import time
import pennylane as qml
from pennylane import numpy as np

from . import qaoa_base as base


class ProblemQAOA:
    """
    QAOA solver for Combinatorial Optimisation Problems (COPs).

    Takes a QUBO matrix and one or more constraint gadgets (VCG instances)
    as structural state preparation circuits, then runs QAOA with a Grover
    or X mixer over the feasible subspace.

    Attributes
    ----------
    qubo : np.ndarray
        QUBO matrix representing the objective function.
    state_prep : list[VCG]
        Constraint gadget circuits (each exposes ``opt_circuit()`` and ``flag_wires``).
    angle_strategy : str
        'QAOA' or 'ma-QAOA'.
    mixer : str
        'Grover' or 'X-Mixer'.
    penalty : list[float]
        Flag-qubit penalty weights (one per flag wire).
    n_layers : int
        Number of QAOA layers.
    samples : int
        Shots for the counts circuit.
    learning_rate : float
        Adam step size.
    steps : int
        Max optimisation steps per restart.
    num_restarts : int
        Number of random restarts.
    overlap_vars : list[tuple[int, int]] or None
        Variable pairs that must be equal across gadgets (penalised).
    overlap_penalty : list[float] or None
        Penalty weight for each overlap pair.
    """

    def __init__(
        self,
        qubo: np.ndarray,
        state_prep,
        angle_strategy: str = "ma-QAOA",
        mixer: str = "Grover",
        penalty: list[float] = [20],
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 100,
        overlap_vars: list[tuple[int, int]] = None,
        overlap_penalty=None,
    ) -> None:
        self.qubo = qubo
        self.state_prep = state_prep
        self.angle_strategy = base.validate_angle_strategy(angle_strategy)
        self.mixer = base.validate_mixer(mixer)
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts
        self.overlap_vars = overlap_vars
        self.overlap_penalty = overlap_penalty
        self.starting_angles_from_qaoa = None

        self.n_x = qubo.shape[0]
        self.var_wires = list(range(self.n_x))
        self.flag_wires = [f for gadget in state_prep for f in gadget.flag_wires]
        self.penalty = penalty * len(self.flag_wires) if len(penalty) == 1 else penalty
        self.all_wires = list(self.var_wires) + list(self.flag_wires)

        self.problem_Ham = self._build_problem_hamiltonian()
        self.num_gamma = base.count_gamma_terms(self.problem_Ham)
        self.num_beta = len(self.all_wires) if self.mixer == "X-Mixer" else 1

    # ------------------------------------------------------------------
    # Hamiltonian
    # ------------------------------------------------------------------

    def _build_problem_hamiltonian(self) -> qml.Hamiltonian:
        """Build QUBO + flag-penalty + optional overlap-penalty Hamiltonian."""
        start = time.time()
        ham = base.build_qubo_hamiltonian(self.qubo, self.var_wires)

        flag_ham = base.build_flag_penalty_hamiltonian(self.flag_wires, self.penalty)
        if flag_ham is not None:
            ham = ham + flag_ham

        if self.overlap_vars is not None:
            overlap_ham = base.build_overlap_hamiltonian(
                self.overlap_vars, self.overlap_penalty
            )
            if overlap_ham is not None:
                ham = ham + overlap_ham

        self.hamiltonian_time = time.time() - start
        return ham

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------

    def problem_circuit(self, angles: np.ndarray) -> None:
        """Apply the QAOA ansatz: state prep, then alternating cost + mixer layers."""
        gammas, betas = base.split_angles(
            angles, self.num_gamma, self.num_beta, self.angle_strategy
        )

        for gadget in self.state_prep:
            gadget.opt_circuit()

        for q in range(self.n_layers):
            base.apply_cost_unitary(self.problem_Ham, gammas, q)
            if self.mixer == "Grover":
                base.apply_grover_mixer(betas[q][0], self.all_wires, self.state_prep)
            elif self.mixer == "X-Mixer":
                base.apply_x_mixer(betas, q, self.all_wires)

    def opt_circuit(self) -> None:
        """Apply the circuit with the previously optimised angles."""
        self.problem_circuit(self.opt_angles)

    def do_evolution_circuit(self, angles: np.ndarray) -> float:
        """Compute <psi| H_problem |psi> for gradient-based optimisation."""
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires))
        def circuit(angles):
            self.problem_circuit(angles)
            return qml.expval(self.problem_Ham)
        return circuit(angles)

    def do_counts_circuit(self, angles=None, shots: int = 1000) -> dict:
        """Sample bitstrings from the circuit (uses opt_angles if angles is None)."""
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            if angles is None:
                self.opt_circuit()
            else:
                self.problem_circuit(angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize_angles(
        self,
        cost_fn,
        maximize: bool = False,
        starting_angles_from_qaoa: np.ndarray = None,
        prev_layer_angles: np.ndarray = None,
    ) -> tuple[float, np.ndarray]:
        """Optimise QAOA angles using Adam with random restarts."""
        starting = (
            starting_angles_from_qaoa
            if starting_angles_from_qaoa is not None
            else self.starting_angles_from_qaoa
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
            starting_angles=starting,
        )
        self.optimize_time = wall_time
        self.opt_angles = best_angles.reshape(
            self.n_layers, self.num_gamma + self.num_beta
        )
        return best_cost, best_angles

    def optimize_angles_with_counts(
        self, starting_angles_from_qaoa: np.ndarray = None
    ) -> tuple[float, np.ndarray]:
        """
        Maximise the probability of the optimal solution via sampling.

        Uses a probs-based objective instead of an expectation value, so
        a custom optimisation loop is needed rather than delegating fully
        to base.run_optimization.
        """
        conv_tol = 1e-6
        opt = qml.AdamOptimizer(stepsize=0.1)
        best_obj_fn = 0
        best_params = None

        def objective(params: np.ndarray) -> float:
            @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires))
            def probs_circuit(params):
                self.problem_circuit(params)
                return qml.probs()
            probs = probs_circuit(params)
            prob_opt = np.sum([
                probs[int(s + '0' * len(self.flag_wires), 2)]
                for s in self.optimal_x
            ])
            return -1 * prob_opt

        start = time.time()
        for _ in range(self.num_restarts):
            if self.starting_angles_from_qaoa is not None:
                init_angles = self.starting_angles_from_qaoa
            else:
                init_angles = base.init_angles(
                    self.n_layers, self.num_gamma, self.num_beta, self.angle_strategy
                )

            angles = init_angles
            new_cost = objective(angles)
            for _ in range(self.steps):
                angles, prev_cost = opt.step_and_cost(objective, angles)
                new_cost = objective(angles)
                if np.abs(new_cost - prev_cost) <= conv_tol:
                    break

            if new_cost < best_obj_fn:
                best_obj_fn = new_cost
                best_params = angles

        self.optimize_time = time.time() - start
        self.opt_angles = best_params
        print(best_params)
        return -1 * best_obj_fn, best_params

    # ------------------------------------------------------------------
    # Resources & feasibility
    # ------------------------------------------------------------------

    def get_circuit_resources(self, opt: bool = False) -> tuple:
        """Estimate shot budget and statistical error for the problem Hamiltonian."""
        est_shots, est_error, group_shots, group_error = (
            base.estimate_hamiltonian_resources(self.problem_Ham)
        )
        return None, est_shots, est_error, group_shots, group_error

    def prob_opt_x_sol_in_counts(self, counts: dict) -> float:
        """Probability that the optimal x solution appears in measured counts."""
        total_counts = sum(counts.values())
        prob_opt = 0.0
        for sol in self.optimal_x:
            sol_key = sol + '0' * len(self.flag_wires)
            if sol_key in counts:
                prob_opt += counts[sol_key] / total_counts
        return prob_opt
