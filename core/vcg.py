"""
vcg.py -- Variational Constraint Gadget (VCG) QAOA.

Builds a Hamiltonian whose ground state encodes constraint satisfaction
(valid states get eigenvalue -1, invalid states +1), then uses QAOA to
prepare that state.

Delegates to:
  - qaoa_base  for circuit primitives, angle utilities, and optimisation.
  - constraint_handler  for constraint parsing.
"""

import os
import re
import time
import itertools as it
import pandas as pd
import pennylane as qml
from pennylane import numpy as np

from . import qaoa_base as base
from . import constraint_handler as ch


def find_in_db(constraints: list, db_path: str,
               n_layers: int = 1, angle_strategy: str = 'ma-QAOA') -> bool:
    """Return True if a matching VCG entry exists in a gadget database pickle.

    Parameters
    ----------
    constraints : list[str]
        Constraint strings (zero-indexed or QUBO-indexed; ConstraintMapper
        handles variable remapping during the search).
    db_path : str
        Path to a pickle produced by collect_vcg_data / ResultsCollector.
    n_layers : int
        Number of QAOA layers the stored gadget must have.
    angle_strategy : str
        Angle strategy ('QAOA' or 'ma-QAOA') the stored gadget must use.

    Returns
    -------
    bool
    """
    if not db_path or not os.path.exists(db_path):
        return False
    try:
        df = pd.read_pickle(db_path)
        if df.empty:
            return False
        mapper = ch.ConstraintMapper(df['constraints'].to_list())
        matched = mapper.map_constraints(constraints)
        if not matched:
            return False
        mask = (
            (df['n_layers'] == n_layers)
            & (df['angle_strategy'] == angle_strategy)
            & df['constraints'].apply(lambda x: set(x) == set(matched))
        )
        return bool(mask.any())
    except Exception:
        return False


def _check_constraint_op(lhs_val: float, op: ch.ConstraintOp, rhs: float) -> bool:
    """Return True if ``lhs_val op rhs`` holds."""
    if op == ch.ConstraintOp.EQ:
        return abs(lhs_val - rhs) < 1e-6
    if op == ch.ConstraintOp.LEQ:
        return lhs_val <= rhs + 1e-6
    if op == ch.ConstraintOp.LT:
        return lhs_val < rhs - 1e-6
    if op == ch.ConstraintOp.GEQ:
        return lhs_val >= rhs - 1e-6
    if op == ch.ConstraintOp.GT:
        return lhs_val > rhs + 1e-6
    return False


class VCG:
    """
    Variational Constraint Gadget (VCG) QAOA.

    Constructs a Hamiltonian whose ground state encodes constraint satisfaction,
    then uses QAOA to prepare that state efficiently.

    Attributes
    ----------
    constraints : list[str]
        Constraint strings.
    flag_wires : list[int]
        Flag qubit wire indices (one per constraint, or one if single_flag).
    angle_strategy : str
        "ma-QAOA" or "QAOA".
    decompose : bool
        Whether to decompose the Hamiltonian into Pauli terms (True) or use
        qml.Hermitian (False).
    single_flag : bool
        Whether to use a single flag qubit for all constraints.
    n_layers : int
        Number of QAOA layers.
    samples : int
        Number of measurement samples.
    learning_rate : float
        Adam step size.
    steps : int
        Max optimisation steps per restart.
    num_restarts : int
        Number of random restarts.
    """

    def __init__(
        self,
        constraints: list[str],
        flag_wires: list[int],
        angle_strategy: str = "ma-QAOA",
        decompose: bool = True,
        single_flag: bool = False,
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 100,
        pre_made: bool = False,
        path: str = None,
    ) -> None:

        self.constraints = constraints
        self.angle_strategy = angle_strategy
        self.decompose = decompose
        self.single_flag = single_flag
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts

        # Parse constraints via constraint_handler
        self.parsed_constraints = ch.parse_constraints(self.constraints)
        all_var_wires = sorted(
            set().union(*(pc.variables for pc in self.parsed_constraints))
        )
        self.var_wires = [sorted(pc.variables) for pc in self.parsed_constraints]
        self.n_x = len(all_var_wires)
        self.n_c = len(self.constraints) if not single_flag else 1
        self.n = self.n_x + self.n_c
        self.flag_wires = flag_wires
        self.all_wires = all_var_wires + list(self.flag_wires)

        if pre_made:
            self.path = path
            self.get_pre_made_data()
        else:
            self.constraint_Ham = self.get_constraint_Hamiltonian()

        self.num_gamma = len(self.constraint_Ham.ops) if self.decompose else 1
        self.num_beta = len(self.all_wires)

    # ------------------------------------------------------------------
    # Pre-made data
    # ------------------------------------------------------------------

    def get_pre_made_data(self) -> None:
        """Load pre-computed Hamiltonian and angles from a pickle file."""
        df = pd.read_pickle(self.path)
        mapper = ch.ConstraintMapper(df['constraints'].to_list())
        matched = mapper.map_constraints(self.constraints)
        df = df[
            (df['n_layers'] == self.n_layers)
            & (df['angle_strategy'] == self.angle_strategy)
            & (
                df['constraints'].apply(
                    lambda x: set(x) == set(matched) if matched else False
                )
            )
        ]
        if not df.empty:
            self.outcomes = df['outcomes'].iloc[0]
            self.constraint_Ham = df['Hamiltonian'].iloc[0].map_wires(
                {i: w for i, w in enumerate(self.all_wires)}
            )
            self.opt_angles = np.array(df['opt_angles'].iloc[0])
        else:
            print(
                "No pre-made data found for the given constraints and parameters."
                " Making new data."
            )
            self.constraint_Ham = self.get_constraint_Hamiltonian()

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    def get_constraint_Hamiltonian(self) -> qml.Hamiltonian:
        """
        Build the constraint Hamiltonian via truth table + Pauli decomposition.

        Valid states (satisfying all constraints) receive eigenvalue -1;
        invalid states receive +1.

        For decompose=True, uses a Walsh-Hadamard transform (WHT) to compute
        Pauli coefficients directly from the diagonal in O(n * 2^n) time and
        O(2^n) space, avoiding the O(4^n) full-matrix construction.
        """
        _, outcomes = self.generate_truth_table(False)
        start = time.time()
        if self.decompose:
            hamiltonian = self._pauli_decompose_diagonal(outcomes)
        else:
            diag = np.diag(outcomes, requires_grad=False)
            hamiltonian = qml.Hermitian(diag, wires=self.all_wires)
        self.hamiltonian_time = time.time() - start
        return hamiltonian

    def _pauli_decompose_diagonal(self, outcomes: list) -> qml.Hamiltonian:
        """Decompose a diagonal {±1} Hamiltonian into Pauli Z terms via WHT.

        The Pauli coefficient for Z-string S is:
            c_S = (1/2^n) * sum_x  outcomes[x] * (-1)^{popcount(x & S)}
        which is the Walsh-Hadamard transform of outcomes, normalised by 2^n.
        Memory: O(2^n).  Time: O(n * 2^n).
        """
        n = self.n
        N = 1 << n  # 2^n
        coeffs = np.array(outcomes, dtype=float)

        # In-place WHT (Hadamard transform) — vectorised over numpy axes
        step = 1
        while step < N:
            view = coeffs.reshape(-1, 2 * step)
            u = view[:, :step].copy()
            v = view[:, step:].copy()
            view[:, :step] = u + v
            view[:, step:] = u - v
            step <<= 1
        coeffs /= N

        # Build Pauli terms — skip near-zero coefficients and the identity
        pauli_coeffs = []
        pauli_ops = []
        for S in range(N):
            if S == 0 or abs(coeffs[S]) < 1e-10:
                continue
            # bit k of S → Z on wire all_wires[k]
            obs = qml.prod(*(
                qml.PauliZ(self.all_wires[k])
                for k in range(n) if S & (1 << k)
            ))
            pauli_coeffs.append(float(coeffs[S]))
            pauli_ops.append(obs)

        return qml.Hamiltonian(np.array(pauli_coeffs), pauli_ops)

    # ------------------------------------------------------------------
    # Truth table
    # ------------------------------------------------------------------

    def generate_truth_table(self, yesPrintTTable: bool = False) -> tuple:
        """
        Build the constraint truth table.

        For each assignment of the decision variables, compute ancilla bits
        (0 = constraint satisfied, 1 = violated) and assign eigenvalue -1
        (valid) or +1 (invalid in the full 2^n state space).
        """
        start = time.time()
        truth_table = []
        valid_bitstrings = set()

        # Map wire index -> position in all_wires
        wires_correct_index = {w: i for i, w in enumerate(self.all_wires)}
        original_bitstrings = list(it.product([0, 1], repeat=self.n_x))

        for original_vars in original_bitstrings:
            ancillae = []
            for pc in self.parsed_constraints:
                lhs_val = pc.constant
                for var_idx, coeff in pc.linear.items():
                    lhs_val += coeff * original_vars[wires_correct_index[var_idx]]
                for (i, j), coeff in pc.quadratic.items():
                    lhs_val += (
                        coeff
                        * original_vars[wires_correct_index[i]]
                        * original_vars[wires_correct_index[j]]
                    )
                is_satisfied = _check_constraint_op(lhs_val, pc.op, pc.rhs)
                ancillae.append(0 if is_satisfied else 1)

            if self.single_flag:
                valid_bitstring = tuple(original_vars) + (int(any(ancillae)),)
            else:
                valid_bitstring = tuple(original_vars) + tuple(ancillae)
            valid_bitstrings.add(valid_bitstring)
            truth_table.append(list(valid_bitstring) + [-1])

        for bitstring in it.product([0, 1], repeat=self.n):
            if bitstring not in valid_bitstrings:
                truth_table.append(list(bitstring) + [1])

        truth_table.sort(key=lambda x: tuple(x[:-1]))

        if yesPrintTTable:
            print("Truth table (original_vars + ancillae + outcome):")
            for row in truth_table:
                print(row)

        outcomes = [row[-1] for row in truth_table]
        self.table_time = time.time() - start
        self.outcomes = outcomes
        return np.array(truth_table), outcomes

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------

    def constraint_circuit(self, angles: np.ndarray) -> None:
        """Apply the constraint QAOA circuit."""
        gammas, betas = base.split_angles(
            angles, self.num_gamma, self.num_beta, self.angle_strategy
        )

        for wire in self.all_wires:
            qml.Hadamard(wires=wire)

        for q in range(self.n_layers):
            if self.decompose:
                base.apply_cost_unitary(self.constraint_Ham, gammas, q)
            else:
                # Diagonal H: exp(-i*gamma*H) is a diagonal unitary whose
                # k-th entry is exp(-i*gamma*outcomes[k]).  DiagonalQubitUnitary
                # works for both statevector and shot-based simulation.
                diag = np.exp(
                    -1j * gammas[q][0] * np.array(self.outcomes, dtype=float)
                )
                qml.DiagonalQubitUnitary(diag, wires=self.all_wires)
            base.apply_x_mixer(betas, q, self.all_wires)

    def opt_circuit(self) -> None:
        """Apply the circuit with the stored optimal angles."""
        self.constraint_circuit(self.opt_angles)

    # ------------------------------------------------------------------
    # QNode wrappers
    # ------------------------------------------------------------------

    def do_evolution_circuit(self, angles: np.ndarray) -> float:
        """Compute <psi|H_constraint|psi> for gradient-based optimisation."""
        dev = qml.device("default.qubit", wires=self.all_wires)

        @qml.qnode(dev)
        def circuit(angles):
            self.constraint_circuit(angles)
            return qml.expval(self.constraint_Ham)

        return circuit(angles)

    def do_counts_circuit(self, probs: bool = False, shots: int = 1000) -> dict:
        """Sample measurement outcomes from the optimised circuit."""
        dev = qml.device("default.qubit", wires=self.all_wires, shots=shots)

        @qml.qnode(dev)
        def circuit():
            self.opt_circuit()
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    def get_circuit_resources(self, opt: bool = False) -> tuple:
        """Estimate gate count and shot budget for the circuit."""
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            if opt:
                self.opt_circuit()
            else:
                angles = base.init_angles(
                    self.n_layers, self.num_gamma, self.num_beta, self.angle_strategy
                )
                self.constraint_circuit(angles)
            return qml.state()

        gate_resources = qml.specs(
            qml.compile(circuit, basis_set=["Hadamard", "CNOT"])
        )()['resources']
        if self.decompose:
            est_shots, est_error, group_shots, group_error = (
                base.estimate_hamiltonian_resources(self.constraint_Ham)
            )
        else:
            est_shots = est_error = group_shots = group_error = 0
        return gate_resources, est_shots, est_error, group_shots, group_error

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize_angles(
        self,
        cost_fn,
        maximize: bool = False,
        starting_angles_from_qaoa: np.ndarray = None,
        prev_layer_angles: np.ndarray = None,
    ) -> tuple:
        """Optimise QAOA angles using Adam with random restarts."""
        starting_angles = None
        if starting_angles_from_qaoa is not None:
            starting_angles = starting_angles_from_qaoa
            if self.angle_strategy == "ma-QAOA":
                starting_angles = base.convert_qaoa_to_ma_angles(
                    starting_angles, self.num_gamma, self.num_beta, self.n_layers
                )

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
            starting_angles=starting_angles,
        )
        self.optimize_time = wall_time
        self.opt_angles = best_angles
        return best_cost, best_angles
