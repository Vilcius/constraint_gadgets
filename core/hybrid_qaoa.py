"""
hybrid_qaoa.py -- HybridQAOA with JAX-jitted optimisation.

Combines structural state preparation (Dicke, Flow, VCG gadgets) with
penalty-based terms for the remaining constraints.  The cost circuit and
Adam optimisation step are JIT-compiled via JAX.

Implementation notes:
  - Device: ``default.qubit`` with JAX interface for autodiff.
  - Grover mixer: ctrl(RZ)+PhaseShift decomposition (compatible with lightning.qubit sampling).
  - Gradient: ``jax.value_and_grad`` (adjoint).
  - Optimizer: ``optax.adam``.
  - Layer loop: Python ``for q in range(n_layers)`` — unrolled at trace time.
  - do_counts_circuit uses lightning.qubit directly (no jit needed for sampling).

Requirements:
  pip install jax optax pennylane
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional

import jax
import jax.numpy as jnp
import optax
import pennylane as qml

# Project modules
from . import qaoa_base as base
from . import constraint_handler as ch
from . import dicke_state_prep as dsp
from . import vcg as vcgmod


class HybridQAOA:
    """
    Hybrid QAOA: structural state preparation + penalty Hamiltonian components.

    Parameters match HybridQAOA exactly so the two are interchangeable in
    run scripts.  See hybrid_qaoa.py for full parameter documentation.
    """

    def __init__(
        self,
        qubo,
        all_constraints: List[ch.ParsedConstraint],
        structural_indices: List[int],
        penalty_indices: List[int],
        penalty_pen: float = 10.0,
        angle_strategy: str = "ma-QAOA",
        mixer: str = "Grover",
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 10,
        decompose: bool = True,
        cqaoa_n_layers: int = 1,
        cqaoa_angle_strategy: str = "ma-QAOA",
        cqaoa_steps: int = 50,
        cqaoa_num_restarts: int = 10,
        dicke_mixer_type: dsp.DickeMixerType = dsp.DickeMixerType.RING_XY,
        gadget_db_path: Optional[str] = None,
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
        self.leq_preps: List[dsp.CardinalityLeqStatePrep] = []
        self.geq_preps: List[dsp.CardinalityGeqSingleStatePrep] = []
        self.flow_preps: List[dsp.FlowStatePrep] = []
        self.gadget_preps: List[vcgmod.VCG] = []

        # Partition structural indices by type
        dicke_idxs = [i for i in structural_indices
                      if ch.is_dicke_compatible(all_constraints[i])]
        leq_idxs = [i for i in structural_indices
                    if ch.is_cardinality_leq_compatible(all_constraints[i])]
        geq_idxs = [i for i in structural_indices
                    if ch.is_cardinality_geq_single_compatible(all_constraints[i])]
        flow_idxs = [i for i in structural_indices
                     if ch.is_flow_compatible(all_constraints[i])]
        gadget_idxs = [i for i in structural_indices
                       if not ch.is_dicke_compatible(all_constraints[i])
                       and not ch.is_cardinality_leq_compatible(all_constraints[i])
                       and not ch.is_cardinality_geq_single_compatible(all_constraints[i])
                       and not ch.is_flow_compatible(all_constraints[i])]

        for idx in dicke_idxs:
            pc = all_constraints[idx]
            prep = dsp.from_parsed_constraint(pc, mixer_type=dicke_mixer_type)
            self.dicke_preps.append(prep)

        for idx in leq_idxs:
            pc = all_constraints[idx]
            prep = dsp.from_cardinality_leq_constraint(pc)
            self.leq_preps.append(prep)

        for idx in geq_idxs:
            pc = all_constraints[idx]
            prep = dsp.from_cardinality_geq_single_constraint(pc)
            self.geq_preps.append(prep)

        for idx in flow_idxs:
            pc = all_constraints[idx]
            prep = dsp.from_flow_constraint(pc, mixer_type=dicke_mixer_type)
            self.flow_preps.append(prep)

        for idx in gadget_idxs:
            pc = all_constraints[idx]
            gadget = _load_vcg_gadget(
                constraints=[pc.raw],
                db_path=gadget_db_path,
                ma_steps=cqaoa_steps,
                ma_restarts=cqaoa_num_restarts,
            )
            self.gadget_preps.append(gadget)

        self.state_prep = (self.dicke_preps + self.leq_preps + self.geq_preps
                           + self.flow_preps + self.gadget_preps)

        # ============================================================
        # 2. Build penalty components (slack variables)
        # ============================================================
        pen_constraints = [all_constraints[i] for i in penalty_indices]
        slack_wire_offset = self.n_x

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
        self.all_wires = self.x_wires + self.slack_wires
        self.n_total = len(self.all_wires)

        self.qubo_ham = base.build_qubo_hamiltonian(self.qubo, self.x_wires)
        self.penalty_ham = (
            self._build_energetic_penalty_hamiltonian() if pen_constraints else None
        )
        self.problem_ham = self._assemble_problem_hamiltonian()

        # --- QAOA parameter counts ---
        self.num_gamma = base.count_gamma_terms(self.problem_ham)

        if self.mixer == "X-Mixer":
            self.num_beta = len(self.all_wires)
        elif self.mixer in ("XY", "Ring-XY"):
            if self.leq_preps or self.geq_preps:
                raise ValueError(
                    "XY/Ring-XY mixer is incompatible with CardinalityLeqStatePrep "
                    "and CardinalityGeqSingleStatePrep. Use mixer='Grover'."
                )
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

        # Compile cost circuit + single Adam step via JAX jit
        self._compiled_cost, self._compiled_step = self._build_compiled_fns()

    # ==================================================================
    # Circuit
    # ==================================================================

    def hybrid_circuit(self, angles) -> None:
        """Full hybrid QAOA circuit."""
        gammas, betas = base.split_angles(
            angles, self.num_gamma, self.num_beta, self.angle_strategy
        )

        # --- structural state preparation (unrolled at trace time) ---
        for prep in self.state_prep:
            prep.opt_circuit()

        # --- initialise slack qubits in |+> ---
        for wire in self.slack_wires:
            qml.Hadamard(wires=wire)

        # --- QAOA layers ---
        problem_ham = self.problem_ham
        mixer = self.mixer
        all_wires = self.all_wires
        state_prep = self.state_prep

        if mixer == "Grover":
            for q in range(self.n_layers):
                base.apply_cost_unitary(problem_ham, gammas, q)
                base.apply_grover_mixer(betas[q][0], all_wires, state_prep)
        elif mixer == "X-Mixer":
            for q in range(self.n_layers):
                base.apply_cost_unitary(problem_ham, gammas, q)
                base.apply_x_mixer(betas, q, all_wires)
        else:  # XY / Ring-XY
            for q in range(self.n_layers):
                base.apply_cost_unitary(problem_ham, gammas, q)
                self._apply_hybrid_xy_mixer(betas[q])

    # ==================================================================
    # Compiled functions
    # ==================================================================

    def _build_compiled_fns(self):
        """
        Build JAX-jitted cost and step functions.

        cost(angles)             -> scalar <H>
        step(angles, opt_state)  -> (new_angles, new_opt_state, cost)
        """
        dev = qml.device("default.qubit", wires=self.all_wires)
        problem_ham = self.problem_ham

        @qml.qnode(dev, interface="jax")
        def cost_circuit(angles):
            self.hybrid_circuit(angles)
            return qml.expval(problem_ham)

        optimizer = optax.adam(self.learning_rate)

        @jax.jit
        def step(angles, opt_state):
            cost, grads = jax.value_and_grad(lambda a: jnp.real(cost_circuit(a)))(angles)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_angles = optax.apply_updates(angles, updates)
            return new_angles, new_opt_state, cost

        compiled_cost = jax.jit(lambda a: jnp.real(cost_circuit(a)))

        return compiled_cost, step

    # ==================================================================
    # Optimisation
    # ==================================================================

    def optimize_angles(
        self,
        cost_fn=None,           # ignored — kept for API compatibility
        maximize: bool = False,
        prev_layer_angles=None,
    ) -> Tuple[float, any]:
        """
        Optimise angles using compiled Adam + optax, with random restarts.

        Optimise angles using Adam (optax) with random restarts.
        Each restart runs the full ``steps`` budget.
        """
        optimizer = optax.adam(self.learning_rate)
        best_cost = float("inf")
        best_angles = None
        sign = -1.0 if maximize else 1.0

        total_params = (self.num_gamma + self.num_beta
                        if self.angle_strategy == "ma-QAOA" else 2)

        # Warm-start: inherit prev layer angles, zero-pad new layer
        if prev_layer_angles is not None:
            new_size = self.n_layers * total_params - prev_layer_angles.size
            starting_angles = jnp.concatenate([
                prev_layer_angles.flatten(),
                jnp.zeros(new_size),
            ]).reshape(self.n_layers, total_params)
        else:
            starting_angles = None

        start_wall = time.time()
        key = jax.random.PRNGKey(int(time.time() * 1e6) % (2 ** 32))

        for restart_idx in range(self.num_restarts):
            key, subkey = jax.random.split(key)
            shape = (self.n_layers, total_params)

            if starting_angles is not None and restart_idx == 0:
                angles = starting_angles
            elif prev_layer_angles is not None:
                new_vals = jax.random.uniform(
                    subkey,
                    (self.n_layers * total_params - prev_layer_angles.size,),
                    minval=-2 * jnp.pi, maxval=2 * jnp.pi,
                )
                angles = jnp.concatenate(
                    [prev_layer_angles.flatten(), new_vals]
                ).reshape(shape)
            else:
                angles = jax.random.uniform(
                    subkey, shape, minval=-2 * jnp.pi, maxval=2 * jnp.pi
                )

            opt_state = optimizer.init(angles)

            # Fixed-length optimisation loop
            for _ in range(self.steps):
                angles, opt_state, _ = self._compiled_step(angles, opt_state)

            final_cost = float(sign * self._compiled_cost(angles))
            if final_cost < best_cost:
                best_cost = final_cost
                best_angles = angles

        self.optimize_time = time.time() - start_wall
        self.opt_angles = best_angles
        return sign * best_cost, best_angles

    def solve(self) -> Tuple[float, dict, any]:
        """Run the full hybrid QAOA: optimise angles, then sample."""
        opt_cost, opt_angles = self.optimize_angles()
        counts = self.do_counts_circuit()
        return opt_cost, counts, opt_angles

    # ==================================================================
    # Sampling (not jitted — shots-based, called once)
    # ==================================================================

    def do_counts_circuit(self, angles=None, shots: int = 1000) -> dict:
        """
        Sample bitstrings from the optimised circuit.

        Sample bitstrings from the optimised circuit using lightning.qubit.
        """
        _angles = self.opt_angles if angles is None else angles

        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            self.hybrid_circuit(_angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    def opt_circuit(self) -> None:
        """Apply the optimised circuit (for use as a subroutine)."""
        self.hybrid_circuit(self.opt_angles)

    # ==================================================================
    # Hybrid XY mixer (Dicke groups get XY, others get RX)
    # ==================================================================

    def _apply_hybrid_xy_mixer(self, beta_row) -> None:
        """
        Apply a hybrid mixer: XY on structured wires, RX on the rest.

        Wire counts and prep list lengths are static so Python loops
        are unrolled at trace time.
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

        remaining_wires = [w for w in self.all_wires if w not in structured_wire_set]
        for wire in remaining_wires:
            qml.RX(beta_row[beta_idx], wires=wire)
            beta_idx += 1

    # ==================================================================
    # Helpers (unchanged from original)
    # ==================================================================

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
    # Hamiltonian assembly
    # ==================================================================

    def _assemble_problem_hamiltonian(self) -> qml.Hamiltonian:
        """Combine QUBO + energetic penalties."""
        ham = self.qubo_ham
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


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper (unchanged from hybrid_qaoa.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_vcg_gadget(
    constraints: list,
    db_path: Optional[str],
    ma_steps: int = 50,
    ma_restarts: int = 10,
) -> vcgmod.VCG:
    """
    Return a ready-to-use VCG for *constraints*.

    If *db_path* points to a vcg_db.pkl that contains a matching entry,
    the pre-trained angles are restored and train() is skipped.  Otherwise
    the gadget is trained from scratch.

    DB lookup uses the normalized form of the constraint (variables remapped
    to x_0, x_1, ... in sorted order) so remapped experiment constraints
    match the pre-trained entries keyed by their canonical form.
    """
    import os
    import pickle

    gadget = vcgmod.VCG(constraints=constraints)

    if db_path and os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
        key = ch.normalize_constraint(constraints[0])
        if key in db:
            entry = db[key]
            gadget.opt_angles = entry['opt_angles']
            gadget.n_layers   = entry['n_layers']
            gadget.ar         = entry['ar']
            gadget.entropy    = entry.get('entropy')
            gadget._single_feasible_bitstring   = entry.get('single_feasible_bitstring')
            gadget._dicke_superposition_weights = entry.get('dicke_superposition_weights')
            if gadget.opt_angles is not None:
                gadget.num_gamma = (len(gadget.constraint_Ham.ops)
                                    if gadget.decompose else 1)
                gadget.num_beta  = gadget.n_x
            return gadget

    gadget.ma_steps    = ma_steps
    gadget.ma_restarts = ma_restarts
    gadget.train(verbose=False)
    return gadget
