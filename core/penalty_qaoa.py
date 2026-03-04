"""
penalty_qaoa.py -- Standard penalty-based QAOA for constrained binary optimisation.

Delegates to:
  - qaoa_base  for Hamiltonian construction, circuit primitives, and optimisation.
  - constraint_handler  for constraint parsing, slack variables, and feasibility.
"""

from __future__ import annotations

import time
from typing import Tuple, Optional

import pennylane as qml
from pennylane import numpy as np

from . import qaoa_base as base
from . import constraint_handler as ch


class PenaltyQAOA:
    """
    Standard penalty-based QAOA for solving COPs with general constraints.

    Handles:
    - Linear constraints with coefficients: e.g., "2*x_0 + 3*x_1 <= 5"
    - Quadratic constraints: e.g., "x_0*x_1 + x_2 == 1"

    The approach:
    1. Add slack variables to convert inequalities to equalities.
    2. Add penalty terms delta * (constraint - b)^2 to the objective.
    3. Run standard QAOA with X-mixer on the augmented problem.

    Attributes
    ----------
    qubo : np.ndarray
        QUBO matrix for the objective function.
    constraints : list[str]
        Constraint strings.
    penalty_param : float
        Penalty weight delta for constraint violations.
    angle_strategy : str
        "QAOA" or "ma-QAOA".
    n_layers : int
        Number of QAOA layers.
    samples : int
        Number of samples to draw from the quantum circuit.
    learning_rate : float
        Adam step size.
    steps : int
        Max optimisation steps per restart.
    num_restarts : int
        Number of random restarts.
    """

    def __init__(
        self,
        qubo: np.ndarray,
        constraints: list[str],
        penalty: float = 10.0,
        angle_strategy: str = "ma-QAOA",
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 10,
    ) -> None:

        self.qubo = qubo
        self.constraints = constraints
        self.penalty_param = penalty
        self.angle_strategy = base.validate_angle_strategy(angle_strategy)
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts

        # Parse constraints
        self.parsed_constraints = ch.parse_constraints(constraints)

        # Original problem variables
        self.n_x = qubo.shape[0]
        self.x_wires = list(range(self.n_x))

        # Slack variables for inequalities
        self.slack_info, self.n_slack = ch.determine_slack_variables(
            self.parsed_constraints, self.n_x
        )
        self.slack_wires = list(range(self.n_x, self.n_x + self.n_slack))

        # All wires (original + slack)
        self.all_wires = self.x_wires + self.slack_wires
        self.n_total = len(self.all_wires)

        # Build Hamiltonians
        self.qubo_Ham = base.build_qubo_hamiltonian(self.qubo, self.x_wires)
        self.penalty_Ham = self._build_penalty_hamiltonian()
        self.full_Ham = self.qubo_Ham + self.penalty_Ham

        # QAOA parameter counts (non-identity terms only)
        self.num_gamma = base.count_gamma_terms(self.full_Ham)
        self.num_beta = len(self.all_wires)  # X-mixer on all qubits

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    def _build_penalty_hamiltonian(self) -> qml.Hamiltonian:
        """Build delta * sum_i (constraint_i - b_i)^2 via qaoa_base."""
        start = time.time()
        ham = base.build_penalty_hamiltonian(
            pen_constraints=self.parsed_constraints,
            slack_infos=self.slack_info,
            delta=self.penalty_param,
            fallback_wire=self.x_wires[0],
        )
        self.hamiltonian_time = time.time() - start
        return ham

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------

    def qaoa_circuit(self, angles: np.ndarray) -> None:
        """Standard QAOA circuit with X-mixer on all wires."""
        gammas, betas = base.split_angles(
            angles, self.num_gamma, self.num_beta, self.angle_strategy
        )

        for wire in self.all_wires:
            qml.Hadamard(wires=wire)

        for q in range(self.n_layers):
            base.apply_cost_unitary(self.full_Ham, gammas, q)
            base.apply_x_mixer(betas, q, self.all_wires)

    # ------------------------------------------------------------------
    # QNode wrappers
    # ------------------------------------------------------------------

    def do_evolution_circuit(self, angles: np.ndarray) -> float:
        """Compute <psi|H_full|psi> for gradient-based optimisation."""
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires))
        def circuit(angles):
            self.qaoa_circuit(angles)
            return qml.expval(self.full_Ham)

        return circuit(angles)

    def do_counts_circuit(
        self, angles: Optional[np.ndarray] = None, shots: int = 1000
    ) -> dict:
        """Sample bitstrings from the circuit."""
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            self.qaoa_circuit(self.opt_angles if angles is None else angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    def get_circuit_resources(self, opt: bool = False) -> Tuple:
        """Estimate shot budget and error for the cost Hamiltonian."""
        est_shots, est_error, group_shots, group_error = (
            base.estimate_hamiltonian_resources(self.full_Ham)
        )
        return None, est_shots, est_error, group_shots, group_error

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def check_feasibility(self, bitstring: str) -> bool:
        """Check if a bitstring satisfies all constraints (ignores slack bits)."""
        return ch.check_feasibility(bitstring, self.parsed_constraints, self.n_x)
