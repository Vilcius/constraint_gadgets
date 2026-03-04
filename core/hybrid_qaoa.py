"""
hybrid_qaoa.py -- Hybrid QAOA for constrained binary optimisation (refactored).

Orchestrates components from qaoa_base, constraint_handler, and dicke_state_prep
rather than building circuits from scratch.

The constraint set C is partitioned into:
  - C_str (structural):  enforced via ConstraintQAOA gadgets or DickeStatePrep
    circuits. The feasible subspace is prepared directly in the initial state.
  - C_pen (penalised):   enforced via quadratic penalty terms in the cost
    Hamiltonian, with slack variables for inequalities.

The full cost Hamiltonian is:

    H_hyb = H_qubo
            + sum_{k in C_str, gadget} (pen_k/2)(I - Z_{flag_k})
            + sum_{k in C_pen} delta_k * (c_k(x) - b_k)^2

The ansatz applies structural state preparation first, then alternating
cost-unitary and mixer layers.

Usage
-----
    from constraint_handler import parse_constraints, partition_constraints
    from hybrid_qaoa import HybridQAOA

    constraints = ["x_0 + x_1 + x_2 == 2", "x_3 + x_4 <= 1"]
    parsed = parse_constraints(constraints)
    str_idx, pen_idx = partition_constraints(parsed, strategy="auto")

    solver = HybridQAOA(
        qubo=Q,
        all_constraints=parsed,
        structural_indices=str_idx,
        penalty_indices=pen_idx,
    )
    cost, counts, angles = solver.solve()
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional

import pennylane as qml
from pennylane import numpy as np

# Project modules
from . import qaoa_base as base
from . import constraint_handler as ch
from . import dicke_state_prep as dsp
from . import vcg


class HybridQAOA:
    """
    Hybrid QAOA: pieces together structural and penalty components.

    Rather than building circuits and Hamiltonians internally, this class
    delegates to:
      - ``qaoa_base``          for Hamiltonian construction, cost unitaries,
                                mixers, optimisation, and resource estimation.
      - ``constraint_handler``  for parsing, classification, slack variables,
                                and feasibility checking.
      - ``dicke_state_prep``    for Dicke-compatible constraints (exact
                                subspace preparation + XY mixer).
      - ``vcg``                 for general structural constraints (gadgets
                                with flag qubits).

    Parameters
    ----------
    qubo : np.ndarray
        QUBO matrix (n_x x n_x).
    all_constraints : list[ParsedConstraint]
        All constraints, already parsed via constraint_handler.parse_constraints.
    structural_indices : list[int]
        Indices into all_constraints to enforce structurally.
    penalty_indices : list[int]
        Indices into all_constraints to enforce via penalty.
    penalty_str : list[float] or None
        Flag-qubit penalty weights for structural (gadget) constraints.
        Broadcast from a single value if needed. Not used for Dicke constraints.
    penalty_pen : float
        Penalty weight delta for energetic penalty terms.
    angle_strategy, mixer, n_layers, steps, num_restarts, learning_rate, samples :
        Standard QAOA hyperparameters (see qaoa_base).
    single_flag, decompose : bool
        Passed to VCG gadgets.
    cqaoa_n_layers, cqaoa_angle_strategy, cqaoa_steps, cqaoa_num_restarts :
        Hyperparameters for pre-training the VCG gadgets (ignored when pre_made=True).
    dicke_mixer_type : DickeMixerType
        Mixer for Dicke-enforced constraints (default: Ring-XY).
    pre_made : bool
        If True, load pre-trained VCG angles from ``gadget_path`` instead of
        optimising them from scratch.
    gadget_path : str or None
        Path to a pickle file produced by a prior VCG run (used when pre_made=True).
    """

    def __init__(
        self,
        qubo: np.ndarray,
        all_constraints: List[ch.ParsedConstraint],
        structural_indices: List[int],
        penalty_indices: List[int],
        penalty_str: Optional[List[float]] = None,
        penalty_pen: float = 10.0,
        angle_strategy: str = "ma-QAOA",
        mixer: str = "Grover",
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 10,
        single_flag: bool = False,
        decompose: bool = True,
        cqaoa_n_layers: int = 1,
        cqaoa_angle_strategy: str = "ma-QAOA",
        cqaoa_steps: int = 50,
        cqaoa_num_restarts: int = 10,
        dicke_mixer_type: dsp.DickeMixerType = dsp.DickeMixerType.RING_XY,
        pre_made: bool = False,
        gadget_path: Optional[str] = None,
    ) -> None:
        # --- validation ---
        self.angle_strategy = base.validate_angle_strategy(angle_strategy)
        self.mixer = base.validate_mixer(mixer)

        # --- store parameters ---
        self.qubo = qubo
        self.all_constraints = all_constraints
        self.structural_indices = structural_indices
        self.penalty_indices = penalty_indices
        self.penalty_pen_param = penalty_pen
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts

        # --- original problem variables ---
        self.n_x = qubo.shape[0]
        self.x_wires = list(range(self.n_x))

        # ============================================================
        # 1. Build structural components (Dicke, Flow, or VCG gadget)
        # ============================================================
        self.dicke_preps: List[dsp.DickeStatePrep] = []
        self.flow_preps: List[dsp.FlowStatePrep] = []
        self.gadget_preps: List[vcg.VCG] = []
        self.flag_wires: List[int] = []

        # Partition structural indices by type
        dicke_idxs = [i for i in structural_indices
                      if ch.is_dicke_compatible(all_constraints[i])]
        flow_idxs = [i for i in structural_indices
                     if ch.is_flow_compatible(all_constraints[i])]
        gadget_idxs = [i for i in structural_indices
                       if not ch.is_dicke_compatible(all_constraints[i])
                       and not ch.is_flow_compatible(all_constraints[i])]

        # Dicke state preps (no flags, exact)
        for idx in dicke_idxs:
            pc = all_constraints[idx]
            prep = dsp.from_parsed_constraint(pc, mixer_type=dicke_mixer_type)
            self.dicke_preps.append(prep)

        # Flow state preps (no flags, Bell-pair + Ring-XY mixer)
        for idx in flow_idxs:
            pc = all_constraints[idx]
            prep = dsp.from_flow_constraint(pc, mixer_type=dicke_mixer_type)
            self.flow_preps.append(prep)

        # One VCG gadget per structural non-Dicke/non-Flow constraint
        flag_start = self.n_x
        for idx in gadget_idxs:
            pc = all_constraints[idx]
            flag_wire = flag_start
            flag_start += 1
            gadget = vcg.VCG(
                constraints=[pc.raw],
                flag_wires=[flag_wire],
                angle_strategy=cqaoa_angle_strategy,
                decompose=decompose,
                single_flag=False,
                n_layers=cqaoa_n_layers,
                steps=cqaoa_steps,
                num_restarts=cqaoa_num_restarts,
                pre_made=pre_made,
                path=gadget_path,
            )
            if not pre_made:
                gadget.optimize_angles(gadget.do_evolution_circuit)
            self.gadget_preps.append(gadget)
            self.flag_wires.append(flag_wire)

        # Unified state_prep list (all objects with opt_circuit())
        self.state_prep = self.dicke_preps + self.flow_preps + self.gadget_preps

        # --- flag penalty weights ---
        if self.flag_wires:
            if penalty_str is None:
                self.penalty_str = [20.0] * len(self.flag_wires)
            elif len(penalty_str) == 1:
                self.penalty_str = penalty_str * len(self.flag_wires)
            else:
                self.penalty_str = penalty_str
        else:
            self.penalty_str = []

        # ============================================================
        # 2. Build penalty components (slack variables)
        # ============================================================
        pen_constraints = [all_constraints[i] for i in penalty_indices]
        slack_wire_offset = self.n_x + len(self.flag_wires)

        if pen_constraints:
            self._slack_infos, self.n_slack = ch.determine_slack_variables(
                pen_constraints, slack_wire_offset
            )
            self.slack_wires = list(
                range(slack_wire_offset, slack_wire_offset + self.n_slack)
            )
        else:
            self._slack_infos = []
            self.n_slack = 0
            self.slack_wires = []

        self._pen_constraints = pen_constraints

        # ============================================================
        # 3. Wire layout & Hamiltonians
        # ============================================================
        self.all_wires = self.x_wires + self.flag_wires + self.slack_wires
        self.n_total = len(self.all_wires)

        # Delegate Hamiltonian building to qaoa_base
        self.qubo_ham = base.build_qubo_hamiltonian(self.qubo, self.x_wires)
        self.flag_penalty_ham = base.build_flag_penalty_hamiltonian(
            self.flag_wires, self.penalty_str
        )
        self.penalty_ham = (
            self._build_energetic_penalty_hamiltonian() if pen_constraints else None
        )
        self.problem_ham = self._assemble_problem_hamiltonian()

        # --- QAOA parameter counts ---
        self.num_gamma = base.count_gamma_terms(self.problem_ham)

        if self.mixer == "X-Mixer":
            self.num_beta = len(self.all_wires)
        elif self.mixer in ("XY", "Ring-XY"):
            # One beta per Dicke group + one per Flow group + one per remaining wire
            structured_wire_set = set()
            for d in self.dicke_preps:
                structured_wire_set.update(d.var_wires)
            for f in self.flow_preps:
                structured_wire_set.update(f.var_wires)
            self.num_beta = len(self.dicke_preps) + len(self.flow_preps) + (
                len(self.all_wires) - len(structured_wire_set)
            )
        else:  # Grover
            self.num_beta = 1

    # ==================================================================
    # Public interface
    # ==================================================================

    def solve(self) -> Tuple[float, dict, np.ndarray]:
        """
        Run the full hybrid QAOA: optimise angles, then sample.

        Returns
        -------
        opt_cost : float
        counts : dict
        opt_angles : np.ndarray
        """
        opt_cost, opt_angles = self.optimize_angles(self.do_evolution_circuit)
        counts = self.do_counts_circuit(shots=self.samples)
        return opt_cost, counts, opt_angles

    def do_evolution_circuit(self, angles: np.ndarray) -> float:
        """Compute <psi| H_hyb |psi> for gradient-based optimisation."""
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires))
        def circuit(angles):
            self.hybrid_circuit(angles)
            return qml.expval(self.problem_ham)
        return circuit(angles)

    def do_counts_circuit(
        self, angles: Optional[np.ndarray] = None, shots: int = 1000
    ) -> dict:
        """Sample bitstrings from the optimised circuit."""
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            self.hybrid_circuit(self.opt_angles if angles is None else angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    def opt_circuit(self) -> None:
        """Apply the optimised circuit (for use as a subroutine)."""
        self.hybrid_circuit(self.opt_angles)

    def hybrid_circuit(self, angles: np.ndarray) -> None:
        """
        Full hybrid QAOA circuit, assembled from components:

        1. Dicke state preps       (exact feasible subspace, no flags)
        2. Constraint gadget preps (approximate, with flags)
        3. Hadamards on slack wires
        4. QAOA layers: cost unitary + mixer
        """
        gammas, betas = base.split_angles(
            angles, self.num_gamma, self.num_beta, self.angle_strategy
        )

        # --- 1 & 2: structural state preparation ---
        for prep in self.state_prep:
            prep.opt_circuit()

        # --- 3: initialise slack qubits in |+> ---
        for wire in self.slack_wires:
            qml.Hadamard(wires=wire)

        # --- 4: QAOA layers ---
        for q in range(self.n_layers):
            # Cost unitary (delegates to qaoa_base)
            base.apply_cost_unitary(self.problem_ham, gammas, q)

            # Mixer (delegates to qaoa_base or local hybrid method)
            if self.mixer == "Grover":
                base.apply_grover_mixer(
                    betas[q][0], self.all_wires, self.state_prep
                )
            elif self.mixer == "X-Mixer":
                base.apply_x_mixer(betas, q, self.all_wires)
            elif self.mixer in ("XY", "Ring-XY"):
                self._apply_hybrid_xy_mixer(betas[q])

    def optimize_angles(
        self,
        cost_fn,
        maximize: bool = False,
        prev_layer_angles: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Optimise QAOA angles using Adam with random restarts."""
        best_cost, best_angles, wall_time = base.run_optimization(
            cost_fn=cost_fn,
            n_layers=self.n_layers,
            num_gamma=self.num_gamma,
            num_beta=self.num_beta,
            angle_strategy=self.angle_strategy,
            steps=self.steps,
            num_restarts=self.num_restarts,
            learning_rate=self.learning_rate,
            maximize=maximize,
            prev_layer_angles=prev_layer_angles,
        )
        self.optimize_time = wall_time
        self.opt_angles = best_angles
        return best_cost, best_angles

    def check_feasibility(self, bitstring: str) -> bool:
        """Check whether a bitstring satisfies ALL constraints."""
        return ch.check_feasibility(bitstring, self.all_constraints, self.n_x)

    def get_circuit_resources(self) -> Tuple:
        """Estimate shot budget and error for the cost Hamiltonian."""
        est_shots, est_error, group_shots, group_error = (
            base.estimate_hamiltonian_resources(self.problem_ham)
        )
        return None, est_shots, est_error, group_shots, group_error

    # ==================================================================
    # Hybrid XY mixer (Dicke groups get XY, others get RX)
    # ==================================================================

    def _apply_hybrid_xy_mixer(self, beta_row: np.ndarray) -> None:
        """
        Apply a hybrid mixer: XY on structured wires, RX on the rest.

        - Dicke groups: Ring-XY on var_wires (preserves Hamming weight).
        - Flow groups: Ring-XY on in_wires + Ring-XY on out_wires (preserves balance).
        - Flag, slack, and remaining wires: RX rotations.
        """
        structured_wire_set = set()
        beta_idx = 0

        for prep in self.dicke_preps:
            prep.mixer_circuit(beta_row[beta_idx])
            structured_wire_set.update(prep.var_wires)
            beta_idx += 1

        for prep in self.flow_preps:
            prep.mixer_circuit(beta_row[beta_idx])
            structured_wire_set.update(prep.var_wires)
            beta_idx += 1

        # RX on remaining wires (flags, slacks, non-structured vars)
        remaining_wires = [w for w in self.all_wires if w not in structured_wire_set]
        for wire in remaining_wires:
            qml.RX(beta_row[beta_idx], wires=wire)
            beta_idx += 1

    # ==================================================================
    # Hamiltonian assembly
    # ==================================================================

    def _assemble_problem_hamiltonian(self) -> qml.Hamiltonian:
        """Combine QUBO + flag penalties + energetic penalties."""
        ham = self.qubo_ham
        if self.flag_penalty_ham is not None:
            ham = ham + self.flag_penalty_ham
        if self.penalty_ham is not None:
            ham = ham + self.penalty_ham
        return ham

    def _build_energetic_penalty_hamiltonian(self) -> qml.Hamiltonian:
        """Build  sum_{k in C_pen} delta * (c_k - b_k)^2  via qaoa_base."""
        start = time.time()
        ham = base.build_penalty_hamiltonian(
            pen_constraints=self._pen_constraints,
            slack_infos=self._slack_infos,
            delta=self.penalty_pen_param,
            fallback_wire=self.all_wires[0],
        )
        self.hamiltonian_time = time.time() - start
        return ham
