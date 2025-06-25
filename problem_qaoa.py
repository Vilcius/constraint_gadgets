import re
import time
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple


class ProblemQAOA:
    """
    Class to takes in a QUBO matrix and constraint state preparation circuits (constraint gadgets) to solve a constrained optimization problem using QAOA.
    The default method for solving the QCBO problem is to use the Grover mixer and the ma-QAOA angle strategy.
    Attributes:
        qubo (np.ndarray): The QUBO matrix representing the objective function.
        state_prep (list[cq.ConstraintQAOA]): List of constraint state preparation circuits.
        angle_strategy (str): The angle strategy to use ('QAOA' or 'ma-QAOA').
        mixer (str): The mixer to use ('Grover' or 'X-Mixer').
        penalty (list[float]): List of penalty values for each constraint.
        n_layers (int): Number of QAOA layers.
        samples (int): Number of samples to draw from the quantum circuit.
        learning_rate (float): Learning rate for the optimizer.
        steps (int): Number of optimization steps.
        num_restarts (int): Number of random restarts for the optimizer.
        overlap_vars (list[tuple[int, int]]): List of tuples indicating which variables should be equal across different constraints.
        overlap_penalty (float): Penalty value for the overlap constraints.
    """

    def __init__(
        self,
        qubo: np.ndarray,
        state_prep,  # : list[cq.ConstraintQAOA],
        angle_strategy: str = "ma-QAOA",
        mixer: str = "Grover",
        penalty: list[float] = [20],
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 100,
        overlap_vars: list[tuple[int, int]] = None,
        overlap_penalty: float = None
    ) -> None:

        self.qubo = qubo
        self.state_prep = state_prep
        self.angle_strategy = self.__error_check_angle_strategy(angle_strategy)
        self.mixer = self.__error_check_mixer(mixer)
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts
        self.overlap_vars = overlap_vars
        self.overlap_penalty = overlap_penalty
        self.starting_angles_from_qaoa = None

        # self.opt_angles = self.get_opt_angles()
        self.n_x = qubo.shape[0]
        self.var_wires = list(range(self.n_x))
        self.flag_wires = [f for qaoa in state_prep for f in qaoa.flag_wires]
        self.penalty = penalty * len(self.flag_wires) if len(penalty) == 1 else penalty
        self.all_wires = list(self.var_wires) + list(self.flag_wires)
        self.qubo_Ham = self.get_qubo_Hamiltonian()
        self.problem_Ham = self.get_problem_Hamiltonian()
        self.num_gamma = self.__get_num_gamma()
        self.num_beta = len(self.all_wires) if self.mixer == "X-Mixer" else 1

    def solve_problem_qaoa(self, n_layers: int = 1) -> tuple[float, np.ndarray, np.ndarray, qml.resource.Resources, float]:
        """
        Runs the QAOA algorithm for the deterministic equivalent of a multi-stage stochastic optimization problem.

        Args:
            n_layers (int): number of QAOA layers to use.

        Returns:
            tuple:
                - float: final expectation value of objective.
                - np.ndarray: array of bit strings sampled from the QAOA circuit.
                - np.ndarray: the optimal parameters found by the optimizer.
                - qml.resource.Resources: gate count of the quantum circuit.
                - float: total time taken to run the algorithm.
        """

        def objective(params: np.ndarray) -> float:
            if self.angle_strategy == "QAOA":
                params = self.__convert_angles(params)
            return self.__do_evolution_circuit(params)

        ar, counts, params, gates, total_time = self.__optimize_circuit(self.n_layers, objective)

        return ar, counts, params, gates, total_time

    def get_problem_Hamiltonian(self) -> qml.Hamiltonian:
        """
        Returns the problem Hamiltonian for the QAOA circuit, which consists of the QUBO Hamiltonian and the constraint flag penalty Hamiltonian.
        The penalty Hamiltonian is of the form (penalty / 2) (I_flag - Z_flag).
        The resulting problem Hamiltonian is H = H_qubo + H_penalty.

        Returns:
            qml.Hamiltonian: The problem Hamiltonian for the QAOA circuit.
        """
        start = time.time()
        pen_coeff = []
        pen_obs = []
        for i, pen in enumerate(self.penalty):
            pen_coeff.append(pen / 2)
            pen_coeff.append(-pen / 2)
            pen_obs.append(qml.Identity(self.flag_wires[i]))
            pen_obs.append(qml.PauliZ(self.flag_wires[i]))

        penalty_ham = qml.Hamiltonian(pen_coeff, pen_obs)
        problem_ham = self.qubo_Ham + penalty_ham
        if self.overlap_vars is not None:
            problem_ham = self.get_overlap_Hamiltoinan(problem_ham)

        end = time.time()
        total_time = end - start
        self.hamiltonian_time = total_time
        return problem_ham

    def get_overlap_Hamiltoinan(self, prob_ham: qml.Hamiltonian) -> qml.Hamiltonian:
        """
        Returns the penalization of the overlap variables to make sure they are the same value, e.g. x_0 = x_1 is penalized as lambda (x_0 - x_1)^2.
        Args:
            prob_ham (qml.Hamiltonian): The problem Hamiltonian without the overlap penalty.
        Returns:
            qml.Hamiltonian: The problem Hamiltonian for the QAOA circuit.
        """
        def _single_overlap_Hamiltonian(wire1, wire2, penalty):
            coeff = [penalty / 2, -penalty / 2]
            obs = [qml.Identity(wire1), qml.PauliZ(wire1) @ qml.PauliZ(wire2)]
            return qml.Hamiltonian(coeff, obs)

        for i, (w1, w2) in enumerate(self.overlap_vars):
            prob_ham += _single_overlap_Hamiltonian(w1, w2, self.overlap_penalty[i])
        return prob_ham

    def get_qubo_Hamiltonian(self) -> qml.Hamiltonian:
        """
        Returns the QUBO Hamiltonian for the based on the QUBO matrix.

        Returns:
            qml.Hamiltonian: The QUBO Hamiltonian for the QAOA circuit.
        """
        qubits = self.var_wires
        nxx = self.n_x

        jj, hh, oo = self.__get_hvals(self.qubo)
        h_coeff = []
        h_obs = []

        for i in range(nxx):
            for j in range(i+1, nxx):
                if jj[i, j] != 0:
                    h_coeff.append(jj[i, j])
                    h_obs.append(qml.PauliZ(qubits[i]) @ qml.PauliZ(qubits[j]))
            if hh[i] != 0:
                h_coeff.append(hh[i])
                h_obs.append(qml.PauliZ(qubits[i]))

        if oo != 0:
            h_coeff.append(oo)
            h_obs.append(qml.Identity(qubits[0]))
        h_coeff = np.array(h_coeff)
        hamiltonian = qml.Hamiltonian(h_coeff, h_obs)
        return hamiltonian

    def __get_hvals(self, Qx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Function to obtain the Ising coefficients from the QUBO coefficients.

        Args:
            Qx (str): The node identifier for which to calculate the Ising coefficients.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - jmat (np.ndarray): The quadratic coefficients matrix.
                - hvec (np.ndarray): The linear coefficients vector.
                - offset (float): The offset for the Hamiltonian.
        """
        nvars = self.n_x
        jmat = np.zeros(shape=(nvars, nvars))
        hvec = np.zeros(nvars)
        quadratic = Qx
        linear = np.diag(quadratic)

        for i in range(nvars):
            # The coefficients for the linear terms
            hvec[i] = hvec[i] - (1/2 * linear[i]
                                 + 1/4 * sum([quadratic[k][i] for k in range(i)])
                                 + 1/4 * sum([quadratic[i][l] for l in range(i+1, nvars)]))

            for j in range(i+1, nvars):
                # The coefficients for the quadratic terms
                jmat[i][j] = jmat[i][j] + quadratic[i][j]/4

        # Correct the offset value
        offset = (np.sum(quadratic)/4 + np.sum(linear)/4)
        return jmat, hvec, offset

    def problem_circuit(self, angles: np.ndarray) -> None:
        """
        Constructs the problem QAOA circuit.

        Args:
            angles (np.ndarray): Parameters for the QAOA circuit.
        """
        if self.angle_strategy == "ma-QAOA":
            gammas = angles[:, :self.num_gamma]
            betas = angles[:, self.num_gamma:]
        else:
            gammas = np.array([angles[:, 0]] * self.num_gamma).T
            betas = np.array([angles[:, 1]] * self.num_beta).T

        for c in self.state_prep:
            c.opt_circuit()

        for q in range(self.n_layers):
            # Cost unitary
            idx = 0
            coeffs, ops = self.problem_Ham.terms()
            for (w, op) in zip(coeffs, ops):
                if re.search(r"^[I]+$", qml.pauli.pauli_word_to_string(op)):
                    continue
                qml.MultiRZ(w * gammas[q][idx], wires=op.wires)
                idx += 1

            # Mixing unitary
            if self.mixer == "Grover":
                for c in self.state_prep[::-1]:
                    qml.adjoint(c.opt_circuit)()
                for i in self.all_wires:
                    qml.PauliX(wires=i)
                qml.ctrl(qml.PhaseShift(betas[q][0]/np.pi, wires=self.all_wires[-1]), control=self.all_wires[:-1])
                for i in self.all_wires:
                    qml.PauliX(wires=i)
                for c in self.state_prep:
                    c.opt_circuit()

            elif self.mixer == "X-Mixer":
                for i, wire in enumerate(self.all_wires):
                    # Custom combination of gates
                    qml.RX(betas[q][i], wires=wire)

    def opt_circuit(self) -> None:
        """
        Constructs the circuit using the optimal angles.
        """
        opt_angles = self.opt_angles

        self.problem_circuit(opt_angles)

    def do_evolution_circuit(self, angles: np.ndarray) -> float:
        """
        Creates the quantum circuit and computes the expectation value of the cost Hamiltonian.

        Args:
            angles (np.ndarray): Parameters for the QAOA circuit.

        Returns:
            float: The expectation value after evolving the quantum circuit.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires))
        def circuit(angles):
            self.problem_circuit(angles)
            return qml.expval(self.problem_Ham)

        return circuit(angles)

    def do_counts_circuit(self, angles=None, shots=1000) -> dict:
        """
        Creates the quantum circuit and returns the distribution of counts or probabilities.

        Args:
            angles (np.ndarray): Parameters for the QAOA circuit.
            shots (int): Number of shots to sample from the quantum circuit.

        Returns:
            dict: The distribution of counts or probabilities from the quantum circuit.
        """
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            if angles is None:
                self.opt_circuit()
            else:
                self.problem_circuit(angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        end = time.time()
        total_time = end - start
        self.count_time = total_time

        return counts

    def __convert_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Converts a list of angles into the MaQAOA format (where the gamma and beta values are repeated).

        Args:
            angles (np.ndarray): List of angles for the QAOA circuit.

        Returns:
            np.ndarray: Converted angles array in the MaQAOA format.
        """
        maqaoa_angles = []
        for gamma, beta in zip(angles[::2], angles[1::2]):
            maqaoa_angles += [gamma] * self.num_gamma
            maqaoa_angles += [beta] * self.num_beta
        return np.array(maqaoa_angles).reshape(-1, 2)

    def optimize_angles(self, cost_fn, maximize: bool = False, starting_angles_from_qaoa: np.ndarray = None, prev_layer_angles: np.ndarray = None) -> tuple[float, np.ndarray]:
        """
        Runs the optimization process for the QAOA algorithm to solve the stochastic optimization problem.

        Args:
            n_layers (int): Number of QAOA layers.
            objective (callable): The objective function to minimize.
            maximize (bool): Whether to maximize the objective
            starting_angles_from_qaoa (np.ndarray): Optional starting angles from a previous QAOA run.
            prev_layer_angles (np.ndarray): Optional angles from a previous layer to use as a starting point.

        Returns:
            tuple:
                - float: final expectation value of objective.
                - np.ndarray: the optimal parameters found by the optimizer.
        """
        conv_tol = 1e-6
        opt = qml.AdamOptimizer(stepsize=0.1)

        best_obj_fn = float('inf')  # Initialize with infinity for comparison
        best_params = None  # Store best angles for optimal cost

        if maximize:
            opt_mult = -1
        else:
            opt_mult = 1

        start = time.time()
        for restart in range(self.num_restarts):
            # In case we know a good starting point
            if self.starting_angles_from_qaoa is not None:
                init_angles = self.starting_angles_from_qaoa
                # reshape angle to match multi-angle QAOA
            elif prev_layer_angles is not None:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.concatenate((prev_layer_angles.flatten(), np.random.uniform(-2*np.pi, 2*np.pi, (self.num_gamma + self.num_beta), requires_grad=True)), axis=0).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.concatenate((prev_layer_angles.flatten, np.random.uniform(-2*np.pi, 2*np.pi, (2), requires_grad=True)), axis=0).reshape(self.n_layers, 2)
            else:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, self.n_layers*(self.num_gamma + self.num_beta), requires_grad=True).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, 2*self.n_layers, requires_grad=True).reshape(self.n_layers, 2)

            angles = init_angles

            # Bookkeeping
            new_cost = opt_mult * cost_fn(angles)

            for i in range(self.steps):
                angles, prev_cost = opt.step_and_cost(cost_fn, angles)

                new_cost = cost_fn(angles)

                conv_prev = np.abs(new_cost - prev_cost)

                # if the difference between the previous cost and the current cost is less than the convergence tolerance
                if conv_prev <= conv_tol:
                    break

            if new_cost < best_obj_fn:
                best_obj_fn = new_cost
                best_params = angles

        end = time.time()
        total_time = end - start
        self.optimize_time = total_time
        self.opt_angles = best_params.reshape(self.n_layers, self.num_gamma + self.num_beta)

        return best_obj_fn, best_params

    def prob_opt_x_sol_in_counts(self, counts: dict) -> float:
        """
        Calculate probability of measuring optimal x solution as they appear in counts.
        Args:
            counts (dict): The distribution of counts from the quantum circuit.
        Returns:
            float: Probability of measuring the optimal solution.
        """
        prob_opt = 0
        total_counts = sum(counts.values())  # Total counts from the dictionary
        for sol in self.optimal_x:
            # Convert binary string to integer
            sol_key = sol+'0'*(len(self.flag_wires))
            if sol_key in counts:
                prob_opt += counts[sol_key] / total_counts
        return prob_opt

    def optimize_angles_with_counts(self, starting_angles_from_qaoa: np.ndarray = None) -> tuple[float, np.ndarray]:
        """
        Optimizes the angles for the QAOA circuit by maximizing the probability of measuring the optimal solution.
        This is done by sampling the circuit by getting its counts an then calculating the probability of measuring the optimal solution.
        Args:
            starting_angles_from_qaoa (np.ndarray): Optional starting angles from a previous QAOA run.
        Returns:
            tuple:
                - float: final expectation value of objective.
                - np.ndarray: the optimal parameters found by the optimizer.
        """
        # set scipy optimizer
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
            prob_opt = np.sum([probs[int(s+'0'*(len(self.flag_wires)), 2)] for s in self.optimal_x])
            return -1 * prob_opt

        start = time.time()
        for restart in range(self.num_restarts):
            # In case we know a good starting point
            if self.starting_angles_from_qaoa is not None:
                init_angles = self.starting_angles_from_qaoa
                # reshape angle to match multi-angle QAOA
            else:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, self.n_layers*(self.num_gamma + self.num_beta), requires_grad=True).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, 2*self.n_layers, requires_grad=True).reshape(self.n_layers, 2)

            angles = init_angles
            new_cost = objective(angles)

            for i in range(self.steps):
                angles, prev_cost = opt.step_and_cost(objective, angles)
                new_cost = objective(angles)
                conv_prev = np.abs(new_cost - prev_cost)

                # if the difference between the previous cost and the current cost is less than the convergence tolerance
                if conv_prev <= conv_tol:
                    break

            if new_cost < best_obj_fn:
                best_obj_fn = new_cost
                best_params = angles

        end = time.time()
        total_time = end - start
        self.optimize_time = total_time
        self.opt_angles = best_params
        print(best_params)
        return -1*best_obj_fn, best_params

    def __error_check_angle_strategy(self, angle_strategy: str) -> str:
        """
        Checks if the provided angle strategy is valid.

        Args:
            angle_strategy(str): The angle strategy to check.

        Returns:
            str: The valid angle strategy.

        Raises:
            ValueError: If the angle strategy is not 'QAOA' or 'ma-QAOA'.
        """
        if angle_strategy in ["QAOA", "ma-QAOA"]:
            return angle_strategy
        else:
            raise ValueError("Angle strategy must be either 'QAOA' or 'ma-QAOA'.")

    def __error_check_mixer(self, mixer: str) -> str:
        """
        Checks if the provided mixer is valid.

        Args:
            mixer(str): The mixer to check.

        Returns:
            str: The valid mixer.

        Raises:
            ValueError: If the mixer is not 'Grover' or 'X-Mixer'.
        """
        if mixer in ["Grover", "X-Mixer"]:
            return mixer
        else:
            raise ValueError("Mixer must be either 'Grover' or 'X-Mixer'.")

    def get_circuit_resources(self, opt: bool = False) -> qml.resource.Resources:
        """
        Returns the gate count of the quantum circuit.
        Args:
            opt (bool): Whether to use the optimal angles or random angles.
        Returns:
            qml.resource.Resources: The gate count of the quantum circuit.
        """
        @qml.qnode(qml.device("default.qubit"))
        def qubo_cost_fn():
            if opt:
                self.opt_circuit()
            else:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, self.n_layers*(self.num_gamma + self.num_beta), requires_grad=True).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, 2*self.n_layers, requires_grad=True).reshape(self.n_layers, 2)
                self.problem_circuit(init_angles)
            return qml.state()

        gate_resources = None
        coeffs, ops = self.problem_Ham.terms()
        est_shots = qml.resource.estimate_shots(coeffs)
        est_error = qml.resource.estimate_error(coeffs)
        group_ops, group_coeffs = qml.pauli.group_observables(ops, coeffs)
        group_shots = qml.resource.estimate_shots(group_coeffs)
        group_error = qml.resource.estimate_error(group_coeffs)
        return gate_resources, est_shots, est_error, group_shots, group_error

    def __get_num_gamma(self) -> int:
        """
        Returns the number of gamma parameters needed for the QAOA circuit.
        Returns:
            int: The number of gamma parameters needed for the QAOA circuit.
        """
        num_gamma = len(self.problem_Ham.ops)  # - 1
        num_gamma = num_gamma - len(self.flag_wires)
        if self.overlap_vars is not None:
            num_gamma -= len(self.overlap_vars)
        return num_gamma
