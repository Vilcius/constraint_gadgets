import re
import time
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, List, Dict
import itertools as it


class PenaltyQAOA:
    """
    Standard penalty-based QAOA for solving QCBOs with general constraints.

    Handles:
    - Linear constraints with coefficients: e.g., "2*x_0 + 3*x_1 <= 5"
    - Quadratic constraints: e.g., "x_0*x_1 + x_2 == 1"
    - General polynomial constraints on binary variables

    This class implements the traditional approach to handling constraints in QAOA:
    1. Add slack variables to convert inequality constraints to equality constraints
    2. Add penalty terms δ * (constraint - b)^2 to the objective function
    3. Run standard QAOA with X-mixer on the augmented problem

    Attributes:
        qubo (np.ndarray): The QUBO matrix representing the objective function.
        constraints (list[str]): List of constraint strings.
        penalty (float): Penalty parameter δ for constraint violations.
        angle_strategy (str): The angle strategy to use ('QAOA' or 'ma-QAOA').
        n_layers (int): Number of QAOA layers.
        samples (int): Number of samples to draw from the quantum circuit.
        learning_rate (float): Learning rate for the optimizer.
        steps (int): Number of optimization steps.
        num_restarts (int): Number of random restarts for the optimizer.
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
        self.angle_strategy = self.__error_check_angle_strategy(angle_strategy)
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts

        # Parse constraints
        self.lhs, self.rhs, self.op = self.parse_constraints(self.constraints)
        self.constraint_structure = [self.parse_constraint_lhs(lhs) for lhs in self.lhs]

        # Original problem variables
        self.n_x = qubo.shape[0]
        self.x_wires = list(range(self.n_x))

        # Add slack variables for inequalities
        self.slack_info = self.determine_slack_variables()
        self.n_slack = sum([info['n_slack'] for info in self.slack_info])
        self.slack_wires = list(range(self.n_x, self.n_x + self.n_slack))

        # All wires (original + slack)
        self.all_wires = self.x_wires + self.slack_wires
        self.n_total = len(self.all_wires)

        # Build Hamiltonians
        self.qubo_Ham = self.get_qubo_Hamiltonian()
        self.penalty_Ham = self.get_penalty_Hamiltonian()
        self.full_Ham = self.qubo_Ham + self.penalty_Ham

        # QAOA parameters
        self.num_gamma = len(self.full_Ham.ops)
        self.num_beta = len(self.all_wires)  # X-mixer on all qubits

    def parse_constraints(self, constraints: list[str]) -> Tuple[List[str], List[float], List[Tuple[str, str]]]:
        """
        Parses the constraint strings and returns the left-hand side, right-hand side, and operator.

        Args:
            constraints (list[str]): List of constraint strings.

        Returns:
            tuple: (lhs, rhs, op) where lhs is list of left-hand sides, rhs is list of right-hand sides,
                   and op is list of (operator_symbol, operator_name) tuples.
        """
        operators = {'==': 'Equals', '<=': 'Leq', '>=': 'Geq', '=': 'Equals', '<': 'LT', '>': 'GT'}

        lhs, rhs, op = [], [], []
        for constraint in constraints:
            for temp_op in operators.keys():
                if temp_op in constraint:
                    temp_lhs, temp_rhs = constraint.split(temp_op)
                    lhs.append(temp_lhs.strip())
                    # RHS can be float or int
                    try:
                        rhs.append(float(temp_rhs.strip()))
                    except ValueError:
                        raise ValueError(f"Right-hand side '{temp_rhs}' is not a valid number")
                    op.append((temp_op, operators[temp_op]))
                    break
        return lhs, rhs, op

    def parse_constraint_lhs(self, lhs: str) -> Dict:
        """
        Parses the left-hand side of a constraint to extract terms, coefficients, and variable indices.

        Handles:
        - Linear terms: "2*x_0", "x_1", "-3*x_2"
        - Quadratic terms: "x_0*x_1", "2*x_0*x_1"
        - Constants: "5", "-3"

        Args:
            lhs (str): Left-hand side of constraint (e.g., "2*x_0 + 3*x_1*x_2 - x_3 + 5")

        Returns:
            dict: {
                'linear': {var_idx: coefficient, ...},
                'quadratic': {(var_i, var_j): coefficient, ...},
                'constant': float
            }
        """
        # Remove all spaces
        lhs = lhs.replace(' ', '')

        # Initialize result
        result = {
            'linear': {},
            'quadratic': {},
            'constant': 0.0
        }

        # Split by + and -, keeping the sign
        # Use regex to split while keeping delimiters
        terms = re.split(r'(?=[+-])', lhs)
        terms = [t for t in terms if t]  # Remove empty strings

        for term in terms:
            term = term.strip()
            if not term:
                continue

            # Check if it's a constant (no x_ in it)
            if 'x_' not in term:
                try:
                    result['constant'] += float(term)
                except ValueError:
                    raise ValueError(f"Cannot parse term '{term}' as constant")
                continue

            # Extract coefficient and variables
            # Pattern: [coefficient*]x_i[*x_j*x_k...]

            # Find all x_i variables
            var_pattern = r'x_(\d+)'
            variables = re.findall(var_pattern, term)
            var_indices = [int(v) for v in variables]

            # Extract coefficient (everything before first x_)
            coeff_match = re.match(r'^([+-]?[\d.]*)\*?x_', term)
            if coeff_match:
                coeff_str = coeff_match.group(1)
                if coeff_str in ['', '+']:
                    coefficient = 1.0
                elif coeff_str == '-':
                    coefficient = -1.0
                else:
                    coefficient = float(coeff_str)
            else:
                # Term starts directly with x_ (e.g., "x_0")
                if term.startswith('-'):
                    coefficient = -1.0
                else:
                    coefficient = 1.0

            # Classify as linear or quadratic
            if len(var_indices) == 1:
                # Linear term
                var_idx = var_indices[0]
                if var_idx in result['linear']:
                    result['linear'][var_idx] += coefficient
                else:
                    result['linear'][var_idx] = coefficient
            elif len(var_indices) == 2:
                # Quadratic term
                var_pair = tuple(sorted(var_indices))
                if var_pair in result['quadratic']:
                    result['quadratic'][var_pair] += coefficient
                else:
                    result['quadratic'][var_pair] = coefficient
            else:
                raise ValueError(f"Terms with more than 2 variables not supported: {term}")

        return result

    def determine_slack_variables(self) -> List[dict]:
        """
        Determines the number and configuration of slack variables needed for each constraint.
        Takes into account coefficients in the constraints.

        Returns:
            list[dict]: List of dicts with keys 'constraint_idx', 'n_slack', 'slack_start_wire', 'operator'
        """
        slack_info = []
        current_slack_wire = self.n_x

        for i, (lhs, rhs, (op_symbol, op_name)) in enumerate(zip(self.lhs, self.rhs, self.op)):
            structure = self.constraint_structure[i]

            # Calculate maximum possible value of LHS
            max_lhs = structure['constant']
            for var_idx, coeff in structure['linear'].items():
                if coeff > 0:
                    max_lhs += coeff  # x_i = 1 contributes +coeff
                # If coeff < 0, x_i = 1 doesn't increase max, so we don't add it
            for var_pair, coeff in structure['quadratic'].items():
                if coeff > 0:
                    max_lhs += coeff  # Both x_i = x_j = 1 contributes +coeff

            # Calculate minimum possible value of LHS
            min_lhs = structure['constant']
            for var_idx, coeff in structure['linear'].items():
                if coeff < 0:
                    min_lhs += coeff  # x_i = 1 contributes -|coeff|
            for var_pair, coeff in structure['quadratic'].items():
                if coeff < 0:
                    min_lhs += coeff

            if op_symbol in ['<=', '<']:
                effective_rhs = rhs if op_symbol == '<=' else rhs - 1
                # Need slack to go from min_lhs to effective_rhs
                # constraint + sum(s_i) = effective_rhs
                # sum(s_i) ranges from 0 to (effective_rhs - min_lhs)
                n_slack = max(0, int(np.ceil(effective_rhs - min_lhs)))
                slack_info.append({
                    'constraint_idx': i,
                    'n_slack': n_slack,
                    'slack_start_wire': current_slack_wire,
                    'operator': 'leq',
                    'rhs': effective_rhs
                })
                current_slack_wire += n_slack
            elif op_symbol in ['>=', '>']:
                effective_rhs = rhs if op_symbol == '>=' else rhs + 1
                # Need slack to go from effective_rhs to max_lhs
                # constraint - sum(s_i) = effective_rhs
                # sum(s_i) ranges from 0 to (max_lhs - effective_rhs)
                n_slack = max(0, int(np.ceil(max_lhs - effective_rhs)))
                slack_info.append({
                    'constraint_idx': i,
                    'n_slack': n_slack,
                    'slack_start_wire': current_slack_wire,
                    'operator': 'geq',
                    'rhs': effective_rhs
                })
                current_slack_wire += n_slack
            else:  # ==, =
                slack_info.append({
                    'constraint_idx': i,
                    'n_slack': 0,
                    'slack_start_wire': None,
                    'operator': 'eq',
                    'rhs': rhs
                })

        return slack_info

    def get_qubo_Hamiltonian(self) -> qml.Hamiltonian:
        """
        Returns the QUBO Hamiltonian for the objective function (acting only on original variables).

        Returns:
            qml.Hamiltonian: The QUBO Hamiltonian.
        """
        qubits = self.x_wires
        n = self.n_x

        jj, hh, oo = self.__get_hvals(self.qubo)
        h_coeff = []
        h_obs = []

        for i in range(n):
            for j in range(i+1, n):
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

    def get_penalty_Hamiltonian(self) -> qml.Hamiltonian:
        """
        Constructs the penalty Hamiltonian δ * sum_i (constraint_i - b_i)^2.

        Handles general polynomial constraints with coefficients:
        - constraint_i = sum_j (c_j * x_j) + sum_{j<k} (c_jk * x_j * x_k) + constant

        For inequalities, adds slack variables:
        - sum(c_j * x_j) + sum(s_i) <= b  becomes  sum(c_j * x_j) + sum(s_i) == b

        Returns:
            qml.Hamiltonian: The penalty Hamiltonian.
        """
        start = time.time()

        penalty_coeffs = []
        penalty_obs = []

        for constraint_idx, info in enumerate(self.slack_info):
            structure = self.constraint_structure[constraint_idx]
            rhs = info['rhs']

            # Build the constraint expression: LHS - rhs = 0
            # LHS = constant + sum(linear terms) + sum(quadratic terms) [+ sum(slacks)]

            # We'll construct (LHS - rhs)^2 using:
            # (sum_i z_i)^2 = sum_i z_i^2 + 2*sum_{i<j} z_i*z_j
            # where z_i represents each term (with its coefficient)

            # Convert to Pauli operators using z_i = (I - Z_i)/2

            # Collect all terms with their coefficients
            terms = []  # List of (coefficient, [wire_indices])

            # Add linear terms
            for var_idx, coeff in structure['linear'].items():
                terms.append((coeff, [var_idx]))

            # Add quadratic terms (these are x_i * x_j, already products)
            for var_pair, coeff in structure['quadratic'].items():
                terms.append((coeff, list(var_pair)))

            # Add slack variables
            if info['n_slack'] > 0:
                slack_wires = list(range(info['slack_start_wire'],
                                         info['slack_start_wire'] + info['n_slack']))
                for slack_wire in slack_wires:
                    if info['operator'] == 'leq':
                        terms.append((1.0, [slack_wire]))  # Add slacks
                    elif info['operator'] == 'geq':
                        terms.append((-1.0, [slack_wire]))  # Subtract slacks

            # Add constant term (constant - rhs)
            constant_term = structure['constant'] - rhs

            # Now expand (sum_i c_i * term_i + constant)^2
            # = sum_i (c_i * term_i)^2 + 2*sum_{i<j} (c_i * term_i)*(c_j * term_j) +
            #   2*constant*sum_i (c_i * term_i) + constant^2

            # 1. Quadratic terms within the penalty: (c_i * term_i)^2
            for coeff, wires in terms:
                # term_i^2: if single var x_i, then x_i^2 = x_i; if product x_i*x_j, need expansion
                if len(wires) == 1:
                    # Linear term: c * x_i, squared: c^2 * x_i^2 = c^2 * x_i
                    # x_i = (I - Z_i)/2
                    penalty_coeffs.append(self.penalty_param * coeff**2 * 0.5)
                    penalty_obs.append(qml.Identity(wires[0]))

                    penalty_coeffs.append(-self.penalty_param * coeff**2 * 0.5)
                    penalty_obs.append(qml.PauliZ(wires[0]))

                elif len(wires) == 2:
                    # Quadratic term: c * x_i * x_j, squared: c^2 * (x_i * x_j)^2 = c^2 * x_i * x_j
                    # x_i * x_j = ((I-Z_i)/2) * ((I-Z_j)/2) = (I - Z_i - Z_j + Z_i*Z_j)/4
                    penalty_coeffs.append(self.penalty_param * coeff**2 * 0.25)
                    penalty_obs.append(qml.Identity(wires[0]))

                    penalty_coeffs.append(-self.penalty_param * coeff**2 * 0.25)
                    penalty_obs.append(qml.PauliZ(wires[0]))

                    penalty_coeffs.append(-self.penalty_param * coeff**2 * 0.25)
                    penalty_obs.append(qml.PauliZ(wires[1]))

                    penalty_coeffs.append(self.penalty_param * coeff**2 * 0.25)
                    penalty_obs.append(qml.PauliZ(wires[0]) @ qml.PauliZ(wires[1]))

            # 2. Cross terms: 2*c_i*c_j * term_i * term_j
            for i in range(len(terms)):
                for j in range(i+1, len(terms)):
                    coeff_i, wires_i = terms[i]
                    coeff_j, wires_j = terms[j]
                    cross_coeff = 2 * coeff_i * coeff_j

                    # Need to compute term_i * term_j
                    if len(wires_i) == 1 and len(wires_j) == 1:
                        # x_i * x_j
                        penalty_coeffs.append(self.penalty_param * cross_coeff * 0.25)
                        penalty_obs.append(qml.Identity(wires_i[0]))

                        penalty_coeffs.append(-self.penalty_param * cross_coeff * 0.25)
                        penalty_obs.append(qml.PauliZ(wires_i[0]))

                        penalty_coeffs.append(-self.penalty_param * cross_coeff * 0.25)
                        penalty_obs.append(qml.PauliZ(wires_j[0]))

                        penalty_coeffs.append(self.penalty_param * cross_coeff * 0.25)
                        penalty_obs.append(qml.PauliZ(wires_i[0]) @ qml.PauliZ(wires_j[0]))

                    elif len(wires_i) == 1 and len(wires_j) == 2:
                        # x_i * (x_j * x_k) = x_i * x_j * x_k (3-way product)
                        # (I-Z_i)/2 * (I-Z_j-Z_k+Z_j*Z_k)/4
                        # This gets complex - need to expand fully
                        all_wires = wires_i + wires_j
                        # For simplicity, use symbolic expansion
                        self._add_multiway_product(penalty_coeffs, penalty_obs,
                                                   self.penalty_param * cross_coeff, all_wires)

                    elif len(wires_i) == 2 and len(wires_j) == 1:
                        # (x_i * x_j) * x_k
                        all_wires = wires_i + wires_j
                        self._add_multiway_product(penalty_coeffs, penalty_obs,
                                                   self.penalty_param * cross_coeff, all_wires)

                    elif len(wires_i) == 2 and len(wires_j) == 2:
                        # (x_i * x_j) * (x_k * x_l) - 4-way product
                        all_wires = wires_i + wires_j
                        self._add_multiway_product(penalty_coeffs, penalty_obs,
                                                   self.penalty_param * cross_coeff, all_wires)

            # 3. Linear terms with constant: 2*constant*sum_i (c_i * term_i)
            for coeff, wires in terms:
                if len(wires) == 1:
                    # 2*constant*c_i*x_i = 2*constant*c_i*(I-Z_i)/2
                    penalty_coeffs.append(self.penalty_param * 2 * constant_term * coeff * 0.5)
                    penalty_obs.append(qml.Identity(wires[0]))

                    penalty_coeffs.append(-self.penalty_param * 2 * constant_term * coeff * 0.5)
                    penalty_obs.append(qml.PauliZ(wires[0]))

                elif len(wires) == 2:
                    # 2*constant*c_i*x_j*x_k
                    penalty_coeffs.append(self.penalty_param * 2 * constant_term * coeff * 0.25)
                    penalty_obs.append(qml.Identity(wires[0]))

                    penalty_coeffs.append(-self.penalty_param * 2 * constant_term * coeff * 0.25)
                    penalty_obs.append(qml.PauliZ(wires[0]))

                    penalty_coeffs.append(-self.penalty_param * 2 * constant_term * coeff * 0.25)
                    penalty_obs.append(qml.PauliZ(wires[1]))

                    penalty_coeffs.append(self.penalty_param * 2 * constant_term * coeff * 0.25)
                    penalty_obs.append(qml.PauliZ(wires[0]) @ qml.PauliZ(wires[1]))

            # 4. Constant squared: constant^2
            if constant_term != 0:
                penalty_coeffs.append(self.penalty_param * constant_term**2)
                penalty_obs.append(qml.Identity(self.all_wires[0]))

        end = time.time()
        self.hamiltonian_time = end - start

        penalty_ham = qml.Hamiltonian(penalty_coeffs, penalty_obs)
        return penalty_ham

    def _add_multiway_product(self, coeffs: list, obs: list, coeff: float, wires: list) -> None:
        """
        Helper function to add multi-way product terms to the Hamiltonian.

        Computes product_{i in wires} (I - Z_i)/2 and adds to coeffs/obs lists.

        Args:
            coeffs (list): List of coefficients to append to
            obs (list): List of observables to append to
            coeff (float): Overall coefficient for this term
            wires (list): List of wire indices involved in the product
        """
        # Remove duplicates (x_i * x_i = x_i for binary variables)
        unique_wires = list(set(wires))
        n_wires = len(unique_wires)

        # Expand product_{i} (I - Z_i)/2 = (1/2^n) * sum_{S subset of wires} (-1)^|S| * product_{i in S} Z_i

        # Iterate over all subsets
        for subset_size in range(n_wires + 1):
            for subset in it.combinations(unique_wires, subset_size):
                # Coefficient: coeff / 2^n * (-1)^subset_size
                term_coeff = coeff / (2**n_wires) * ((-1)**subset_size)

                if len(subset) == 0:
                    # Identity term
                    coeffs.append(term_coeff)
                    obs.append(qml.Identity(unique_wires[0]))
                elif len(subset) == 1:
                    # Single Z
                    coeffs.append(term_coeff)
                    obs.append(qml.PauliZ(subset[0]))
                else:
                    # Multiple Z's
                    coeffs.append(term_coeff)
                    pauli_term = qml.PauliZ(subset[0])
                    for wire in subset[1:]:
                        pauli_term = pauli_term @ qml.PauliZ(wire)
                    obs.append(pauli_term)

    def __get_hvals(self, Qx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Function to obtain the Ising coefficients from the QUBO coefficients.

        Args:
            Qx (np.ndarray): The QUBO matrix.

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
            hvec[i] = hvec[i] - (1/2 * linear[i]
                                 + 1/4 * sum([quadratic[k][i] for k in range(i)])
                                 + 1/4 * sum([quadratic[i][l] for l in range(i+1, nvars)]))

            for j in range(i+1, nvars):
                jmat[i][j] = jmat[i][j] + quadratic[i][j]/4

        offset = (np.sum(quadratic)/4 + np.sum(linear)/4)
        return jmat, hvec, offset

    def qaoa_circuit(self, angles: np.ndarray) -> None:
        """
        Constructs the standard QAOA circuit with X-mixer.

        Args:
            angles (np.ndarray): Parameters for the QAOA circuit.
        """
        if self.angle_strategy == "ma-QAOA":
            gammas = angles[:, :self.num_gamma]
            betas = angles[:, self.num_gamma:]
        else:
            gammas = np.array([angles[:, 0]] * self.num_gamma).T
            betas = np.array([angles[:, 1]] * self.num_beta).T

        # Initial state: |+>^n (equal superposition of all states)
        for wire in self.all_wires:
            qml.Hadamard(wires=wire)

        for q in range(self.n_layers):
            # Cost unitary
            idx = 0
            coeffs, ops = self.full_Ham.terms()
            for (w, op) in zip(coeffs, ops):
                if re.search(r"^[I]+$", qml.pauli.pauli_word_to_string(op)):
                    continue
                qml.MultiRZ(w * gammas[q][idx], wires=op.wires)
                idx += 1

            # Mixing unitary (standard X-mixer)
            for i, wire in enumerate(self.all_wires):
                qml.RX(betas[q][i], wires=wire)

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
            self.qaoa_circuit(angles)
            return qml.expval(self.full_Ham)

        return circuit(angles)

    def do_counts_circuit(self, angles=None, shots=1000) -> dict:
        """
        Creates the quantum circuit and returns the distribution of counts.

        Args:
            angles (np.ndarray): Parameters for the QAOA circuit. If None, uses optimal angles.
            shots (int): Number of shots to sample from the quantum circuit.

        Returns:
            dict: The distribution of counts from the quantum circuit.
        """
        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            if angles is None:
                self.qaoa_circuit(self.opt_angles)
            else:
                self.qaoa_circuit(angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        end = time.time()
        self.count_time = end - start

        return counts

    def optimize_angles(self, cost_fn, maximize: bool = False, prev_layer_angles: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """
        Runs the optimization process for the QAOA algorithm.

        Args:
            cost_fn (callable): The cost function to minimize.
            maximize (bool): Whether to maximize the objective.
            prev_layer_angles (np.ndarray): Optional angles from a previous layer for warm-starting.

        Returns:
            tuple:
                - float: final expectation value of objective.
                - np.ndarray: the optimal parameters found by the optimizer.
        """
        conv_tol = 1e-6
        opt = qml.AdamOptimizer(stepsize=0.1)

        best_obj_fn = float('inf')
        best_params = None

        if maximize:
            opt_mult = -1
        else:
            opt_mult = 1

        start = time.time()
        for restart in range(self.num_restarts):
            if prev_layer_angles is not None:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.concatenate((prev_layer_angles.flatten(),
                                                  np.random.uniform(-2*np.pi, 2*np.pi,
                                                                    (self.num_gamma + self.num_beta),
                                                                    requires_grad=True)),
                                                 axis=0).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.concatenate((prev_layer_angles.flatten(),
                                                  np.random.uniform(-2*np.pi, 2*np.pi, (2),
                                                                    requires_grad=True)),
                                                 axis=0).reshape(self.n_layers, 2)
            else:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi,
                                                    self.n_layers*(self.num_gamma + self.num_beta),
                                                    requires_grad=True).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, 2*self.n_layers,
                                                    requires_grad=True).reshape(self.n_layers, 2)

            angles = init_angles
            new_cost = opt_mult * cost_fn(angles)

            for i in range(self.steps):
                angles, prev_cost = opt.step_and_cost(cost_fn, angles)
                new_cost = cost_fn(angles)
                conv_prev = np.abs(new_cost - prev_cost)

                if conv_prev <= conv_tol:
                    break

            if new_cost < best_obj_fn:
                best_obj_fn = new_cost
                best_params = angles

        end = time.time()
        self.optimize_time = end - start
        self.opt_angles = best_params.reshape(self.n_layers, self.num_gamma + self.num_beta) if self.angle_strategy == "ma-QAOA" else best_params

        return best_obj_fn, best_params

    def check_feasibility(self, bitstring: str) -> bool:
        """
        Check if a bitstring satisfies all constraints (using only x variables, ignoring slack).

        Args:
            bitstring (str): Binary string of length n_total (includes slack variables)

        Returns:
            bool: True if all constraints are satisfied, False otherwise
        """
        # Extract only the x variables (first n_x bits)
        x_bits = bitstring[:self.n_x]

        for i, (lhs, rhs, (op_symbol, op_name)) in enumerate(zip(self.lhs, self.rhs, self.op)):
            structure = self.constraint_structure[i]

            # Evaluate LHS using the parsed structure
            eval_lhs = structure['constant']

            # Add linear terms
            for var_idx, coeff in structure['linear'].items():
                eval_lhs += coeff * int(x_bits[var_idx])

            # Add quadratic terms
            for (var_i, var_j), coeff in structure['quadratic'].items():
                eval_lhs += coeff * int(x_bits[var_i]) * int(x_bits[var_j])

            # Check constraint
            if op_symbol in ['==', '=']:
                if not np.isclose(eval_lhs, rhs, atol=1e-6):
                    return False
            elif op_symbol == '<=':
                if eval_lhs > rhs + 1e-6:
                    return False
            elif op_symbol == '<':
                if eval_lhs >= rhs - 1e-6:
                    return False
            elif op_symbol == '>=':
                if eval_lhs < rhs - 1e-6:
                    return False
            elif op_symbol == '>':
                if eval_lhs <= rhs + 1e-6:
                    return False

        return True

    def get_circuit_resources(self, opt: bool = False) -> Tuple:
        """
        Returns the resources required for the circuit.

        Args:
            opt (bool): Whether to use the optimal angles.

        Returns:
            tuple: (gate_resources, est_shots, est_error, group_shots, group_error)
        """
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            if opt:
                self.qaoa_circuit(self.opt_angles)
            else:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi,
                                                    self.n_layers*(self.num_gamma + self.num_beta)).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, 2*self.n_layers,
                                                    requires_grad=True).reshape(self.n_layers, 2)
                self.qaoa_circuit(init_angles)
            return qml.state()

        gate_resources = None
        coeffs, ops = self.full_Ham.terms()
        est_shots = qml.resource.estimate_shots(coeffs)
        est_error = qml.resource.estimate_error(coeffs)
        group_ops, group_coeffs = qml.pauli.group_observables(ops, coeffs)
        group_shots = qml.resource.estimate_shots(group_coeffs)
        group_error = qml.resource.estimate_error(group_coeffs)

        return gate_resources, est_shots, est_error, group_shots, group_error

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
