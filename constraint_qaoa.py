import re
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import itertools as it
import time


class ConstraintQAOA:
    """
    A class to implement the Constraint QAOA or Constraint Gadget algorithm.
    This class constructs a QAOA circuit to handle constraints in optimization problems.
    The gadget takes a list of constraints, flag wires, and various parameters to set up the QAOA circuit.
    Attributes:
        constraints (list[str]): A list of constraint strings.
        flag_wires (list[int]): A list of wires to be used as flag qubits.
        angle_strategy (str): The strategy for angle optimization ("ma-QAOA" or "QAOA").
        decompose (bool): Whether to decompose the Hamiltonian into Pauli terms.
        single_flag (bool): Whether to use a single flag qubit for all constraints.
        n_layers (int): The number of QAOA layers.
        samples (int): The number of samples for measurement.
        learning_rate (float): The learning rate for the optimizer.
        steps (int): The number of optimization steps.
        num_restarts (int): The number of random restarts for optimization.
        pre_made (bool): Whether to use pre-made data for the Hamiltonian and angles.
        path (str): The path to the pre-made data file.
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

        self.lhs, self.rhs, self.op = self.parse_constraint(self.constraints)
        self.var_wires = [[int(x) for x in re.findall(r"x_*(\d+)", c)] for c in self.lhs]
        self.n_x = len(list(set(w for ww in self.var_wires for w in ww)))
        self.n_c = len(self.constraints) if not single_flag else 1
        self.n = self.n_x + self.n_c
        self.flag_wires = flag_wires
        self.all_wires = list(set(w for ww in self.var_wires for w in ww)) + list(self.flag_wires)
        if pre_made:
            self.path = path
            self.get_pre_made_data()
        else:
            self.constraint_Ham = self.get_constraint_Hamiltonian()
        self.num_gamma = len(self.constraint_Ham.ops) if self.decompose else 1
        self.num_beta = len(self.all_wires)

    def get_pre_made_data(self) -> None:
        """
        Loads the pre-made data from the given path.

        Args:
            path (str): The path to the pre-made data.
        """
        df = pd.read_pickle(self.path)
        mapper = ConstraintMapper(df['constraints'].to_list())
        df = df[(df['n_layers'] == self.n_layers) & (df['angle_strategy'] == self.angle_strategy) & (df['constraints'].apply(lambda x: set(x) == set(mapper.map_constraints(self.constraints))))]
        if not df.empty:
            self.outcomes = df['outcomes'].iloc[0]
            self.constraint_Ham = df['Hamiltonian'].iloc[0].map_wires({i: w for i, w in enumerate(self.all_wires)})
            self.opt_angles = np.array(df['opt_angles'].iloc[0])
        else:
            print("No pre-made data found for the given constraints and parameters. Making new data.")
            self.constraint_Ham = self.get_constraint_Hamiltonian()

    def parse_constraint(self, constraint: str) -> tuple[str, int, str]:
        """
        Parses the constraint string and returns the left-hand side, right-hand side, and operator.
        Args:
            constraint (str): The constraint string to parse.

        Returns:
            tuple[str, int, str]: The left-hand side, right-hand side, and operator of the constraint.
        """
        operators = {'==': 'Equals', '<=': 'Leq', '>=': 'Geq', '=': 'Equals', '<': 'LT', '>': 'GT'}

        lhs, rhs, op = [], [], []
        for constraint in self.constraints:
            for temp_op in operators.keys():
                if temp_op in constraint:
                    temp_lhs, temp_rhs = constraint.split(temp_op)
                    lhs.append(temp_lhs)
                    rhs.append(temp_rhs)
                    op.append([temp_op, operators[temp_op]])
                    break
        return lhs, rhs, op

    def get_constraint_Hamiltonian(self) -> qml.Hamiltonian:
        """
        Creates the constraint Hamiltonian. If the constraint has been optimized before, it will load the Hamiltonian from a file. Otherwise, it will construct the Hamiltonian.
        """
        true_table, outcomes = self.generate_truth_table(False)
        diag = np.diag(outcomes, requires_grad=False)
        start = time.time()
        if self.decompose:
            hamiltonian = qml.pauli_decompose(diag, hide_identity=True, wire_order=self.all_wires)
        else:
            hamiltonian = qml.Hermitian(diag, wires=self.all_wires)
        end = time.time()
        total_time = end - start
        self.hamiltonian_time = total_time
        return hamiltonian

    def constraint_circuit(self, angles: np.ndarray) -> None:
        """
        Constructs the constraint QAOA circuit.

        Args:
            angles (np.ndarray): Parameters for the QAOA circuit.
        """
        if self.angle_strategy == "ma-QAOA":
            gammas = angles[:, :self.num_gamma]
            betas = angles[:, self.num_gamma:]
        else:
            gammas = np.array([angles[:, 0]] * self.num_gamma).T
            betas = np.array([angles[:, 1]] * self.num_beta).T

        for wire in self.all_wires:
            qml.Hadamard(wires=wire)

        for q in range(self.n_layers):
            # Cost unitary
            if self.decompose:
                idx = 0
                coeffs, ops = self.constraint_Ham.terms()
                for (w, op) in zip(coeffs, ops):
                    if re.search(r"^[I]+$", qml.pauli.pauli_word_to_string(op)):
                        continue
                    qml.MultiRZ(w * gammas[q][idx], wires=op.wires)
                    idx += 1
            else:
                hamiltonian = self.constraint_Ham
                qml.evolve(hamiltonian, coeff=gammas[q][0])

            # Mixing unitary
            for i, wire in enumerate(self.all_wires):
                qml.RX(betas[q][i], wires=wire)

    def opt_circuit(self) -> None:
        """
        Constructs the circuit using the optimal angles.
        """
        opt_angles = self.opt_angles

        self.constraint_circuit(opt_angles)

    def do_evolution_circuit(self, angles: np.ndarray) -> float:
        """
        Creates the quantum circuit and computes the expectation value of the cost Hamiltonian.

        Args:
            angles (np.ndarray): Parameters for the QAOA circuit.

        Returns:
            float: The expectation value after evolving the quantum circuit.
        """

        if self.decompose:
            dev = qml.device("default.qubit", wires=self.all_wires)
        else:
            dev = qml.device("default.qubit", wires=self.all_wires)

        @qml.qnode(dev)
        def circuit(angles):
            self.constraint_circuit(angles)
            hamiltonian = self.constraint_Ham
            return qml.expval(hamiltonian)

        return circuit(angles)

    def do_counts_circuit(self, probs=False, shots=1000) -> dict:
        """
        Creates the quantum circuit and returns the distribution of measurement outcomes.

        Args:
            probs (bool): Whether to return probabilities instead of counts.
            shots (int): The number of shots to use for sampling.

        Returns:
            dict: The distribution of measurement outcomes.
        """
        if self.decompose:
            dev = qml.device("default.qubit", wires=self.all_wires, shots=shots)
        else:
            dev = qml.device("default.qubit", wires=self.all_wires, shots=shots)

        @qml.qnode(dev)
        def circuit():
            self.opt_circuit()
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        end = time.time()
        total_time = end - start
        self.count_time = total_time

        return counts

    def get_circuit_resources(self, opt: bool = False) -> qml.resource.Resources:
        """
        Returns the resources required for the circuit.

        Args:
            opt (bool): Whether to use the optimal angles.

        Returns:
            qml.resource.Resources: The resources required for the circuit.
        """
        @qml.qnode(qml.device("default.qubit"))
        def qubo_cost_fn():
            if opt:
                self.opt_circuit()
            else:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, self.n_layers*(self.num_gamma + self.num_beta)).reshape(self.n_layers, self.num_gamma + self.num_beta)
                else:
                    init_angles = np.random.uniform(-2*np.pi, 2*np.pi, 2*self.n_layers, requires_grad=True).reshape(self.n_layers, 2)
                self.constraint_circuit(init_angles)
            return qml.state()

        gate_resources = qml.specs(qml.compile(qubo_cost_fn, basis_set=["Hadamard", "CNOT"]))()['resources']
        if self.decompose:
            coeffs, ops = self.constraint_Ham.terms()
            est_shots = qml.resource.estimate_shots(coeffs)
            est_error = qml.resource.estimate_error(coeffs)
            group_ops, group_coeffs = qml.pauli.group_observables(ops, coeffs)
            group_shots = qml.resource.estimate_shots(group_coeffs)
            group_error = qml.resource.estimate_error(group_coeffs)
        else:
            est_shots = 0
            est_error = 0
            group_shots = 0
            group_error = 0
        return gate_resources, est_shots, est_error, group_shots, group_error

    def __convert_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Converts a list of angles into the MaQAOA format (where the gamma and beta values are repeated).

        Args:
            angles (np.ndarray): List of angles for the QAOA circuit.

        Returns:
            np.ndarray: Converted angles array in the MaQAOA format.
        """
        maqaoa_angles = []
        if self.n_layers == 1:
            # If there is only one layer, we can just repeat the angles
            for gamma, beta in angles:
                maqaoa_angles += [gamma] * self.num_gamma
                maqaoa_angles += [beta] * self.num_beta
            return np.array(maqaoa_angles).reshape(1, self.num_gamma + self.num_beta)
        else:
            for i in range(self.n_layers):
                for gamma, beta in angles[i]:
                    maqaoa_angles += [gamma] * self.num_gamma
                    maqaoa_angles += [beta] * self.num_beta
            return np.array(maqaoa_angles).reshape(self.n_layers, self.num_gamma + self.num_beta)

    def generate_truth_table(self, yesPrintTTable: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the truth table for the constraint.

        Args:
            yesPrintTTable (bool): Whether to print the truth table.

        Returns:
            tuple[np.ndarray, np.ndarray]: The truth table and the outcomes.
        """
        start = time.time()
        # Prepare to store the truth table
        truth_table = []
        valid_bitstrings = set()

        # Create all possible bitstrings for original vars
        original_bitstrings = list(it.product([0, 1], repeat=self.n_x))

        for original_vars in original_bitstrings:
            # Check each constraint and create the corresponding ancilla bits
            ancillae = []
            # Create a dictionary to map the wire names to their indices
            wires_correct_index = {w: i for i, w in enumerate(self.all_wires)}
            for c in range(len(self.constraints)):
                # Evaluate the left-hand side of the constraint
                var_dict = {f'x_{x}': original_vars[wires_correct_index[x]] for x in self.var_wires[c]}
                eval_lhs = eval(self.lhs[c], {}, var_dict)

                # Check satisfaction based on the type of constraint
                is_satisfied = (eval(f'{eval_lhs} {self.op[c][0]} {self.rhs[c]}'))

                ancillae.append(0 if is_satisfied else 1)

            # Append the valid bitstring (original vars + ancillae) with outcome 0
            if self.single_flag:
                valid_bitstring = tuple(original_vars) + tuple([int(any(ancillae))])
            else:
                valid_bitstring = tuple(original_vars) + tuple(ancillae)
            valid_bitstrings.add(valid_bitstring)
            truth_table.append(list(valid_bitstring) + [-1])

        # Generate all possible bitstrings for total_vars
        all_bitstrings = list(it.product([0, 1], repeat=self.n))

        # Mark the remaining bitstrings with outcome 1
        for bitstring in all_bitstrings:
            if bitstring not in valid_bitstrings:
                invalid_bitstring = list(bitstring) + [1]
                truth_table.append(invalid_bitstring)

        # Sort the truth table based on the bitstrings
        truth_table.sort(key=lambda x: tuple(x[:-1]))

        if yesPrintTTable:
            print("Truth table (original_vars + ancillae + outcome):")
            for row in truth_table:
                print(row)

        # Separate outcomes from the truth table
        outcomes = [row[-1] for row in truth_table]
        end = time.time()
        total_time = end - start
        self.table_time = total_time
        self.outcomes = outcomes

        return np.array(truth_table), outcomes

    def optimize_angles(self, cost_fn, maximize: bool = False, starting_angles_from_qaoa: np.ndarray = None, prev_layer_angles: np.ndarray = None) -> tuple[float, np.ndarray]:
        """
            Optimize the angles for the QAOA circuit.

            Args:
                cost_fn (function): The cost function to optimize.
                maximize (bool): Whether to maximize the cost function.
                starting_angles_from_qaoa (np.ndarray): The starting angles for the optimization.
                prev_layer_angles (np.ndarray): The angles from the previous layer for layer-wise optimization.
            Returns:
                tuple[float, np.ndarray]: The optimal cost and the optimal angles.
        """
        conv_tol = 1e-6
        opt = qml.AdamOptimizer(stepsize=0.1)

        best_cost = float('inf')  # Initialize with infinity for comparison
        best_angles = None  # Store best angles for optimal cost

        if maximize:
            opt_mult = -1
        else:
            opt_mult = 1

        start = time.time()
        for restart in range(self.num_restarts):
            # In case we know a good starting point
            if starting_angles_from_qaoa is not None:
                init_angles = starting_angles_from_qaoa
                init_angles = self.__convert_angles(init_angles) if self.angle_strategy == "ma-QAOA" else init_angles
                # reshape angle to match multi-angle QAOA
            elif prev_layer_angles is not None:
                if self.angle_strategy == "ma-QAOA":
                    init_angles = np.concatenate((prev_layer_angles, np.random.uniform(-2*np.pi, 2*np.pi, self.n_layers*(self.num_gamma + self.num_beta - prev_layer_angles.shape[1]), requires_grad=True).reshape(self.n_layers, self.num_gamma + self.num_beta - prev_layer_angles.shape[1])), axis=1)
                else:
                    init_angles = np.concatenate((prev_layer_angles, np.random.uniform(-2*np.pi, 2*np.pi, 2*self.n_layers - prev_layer_angles.shape[1], requires_grad=True).reshape(self.n_layers, 2 - prev_layer_angles.shape[1])), axis=1)
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

            if new_cost < best_cost:
                best_cost = new_cost
                best_angles = angles

        end = time.time()
        total_time = end - start
        self.optimize_time = total_time
        self.opt_angles = best_angles

        return best_cost, best_angles


class ConstraintMapper:
    """
    A class to map a set of constraints to a pre-existing set of constraints in a dataframe.
    """

    def __init__(self, constraints_in_df):
        self.constraints_in_df = constraints_in_df

    def normalize_constraint(self, constraint: str) -> str:
        """
        Normalizes a constraint by removing spaces, sorting terms within the constraint, and standardizing the operator.
        Args:
            constraint (str): The constraint to normalize.
        Returns:
            str: The normalized constraint.
        """
        # Remove spaces and sort terms within the constraint
        constraint = re.sub(r'\s+', '', constraint)
        operator = re.search(r'(==|<=|>=|=|<|>)', constraint).group(0)
        lhs, rhs = constraint.split(operator)
        terms = sorted(re.split(r'(\+)', lhs))
        normalized_lhs = ' '.join(terms)
        return f"{normalized_lhs}{operator}{rhs}"

    def normalize_constraints(self, constraints: list[str]) -> list[str]:
        """
        Normalizes a list of constraints.
        Args:
            constraints (list[str]): The list of constraints to normalize.
        Returns:
            list[str]: The list of normalized constraints.
        """
        return [self.normalize_constraint(c) for c in constraints]

    def map_constraints(self, input_constraints: list[str]) -> list[str] | None:
        """
        Maps the input constraints to a set of constraints in the dataframe.
        Args:
            input_constraints (list[str]): The input constraints to map.
        Returns:
            list[str] | None: The mapped constraints if a match is found, otherwise None.
        """
        normalized_input = self.normalize_constraints(input_constraints)
        for constraints in self.constraints_in_df:
            normalized_constraints = self.normalize_constraints(constraints)
            if self.match_constraints(normalized_input, normalized_constraints):
                return constraints
        return None

    def match_constraints(self, input_constraints: list[str], df_constraints: list[str]) -> bool:
        """
        Checks if two sets of constraints match, allowing for permutations of variables and constraints.
        Args:
            input_constraints (list[str]): The input constraints to check.
            df_constraints (list[str]): The constraints from the dataframe to check against.
        Returns:
            bool: True if the constraints match, otherwise False.
        """
        input_vars = sorted(set(re.findall(r'x_\d+', ' '.join(input_constraints))))
        df_vars = sorted(set(re.findall(r'x_\d+', ' '.join(df_constraints))))

        if len(input_vars) != len(df_vars):
            return False

        # Generate all possible mappings of input_vars to df_vars
        for perm in it.permutations(df_vars):
            var_map = {input_var: perm[i] for i, input_var in enumerate(input_vars)}
            mapped_input_constraints = [
                re.sub(r'x_\d+', lambda m: var_map[m.group(0)], constraint)
                for constraint in input_constraints
            ]

            # Check all permutations of the constraints themselves
            for perm_constraints in it.permutations(mapped_input_constraints):
                if self.check_permutations(perm_constraints, df_constraints):
                    return True

        return False

    def check_permutations(self, perm_constraints: list[str], df_constraints: list[str]) -> bool:
        """
        Checks if two sets of constraints match, allowing for permutations of variables within each constraint.
        Args:
            perm_constraints (list[str]): The permuted input constraints to check.
            df_constraints (list[str]): The constraints from the dataframe to check against.
        Returns:
            bool: True if the constraints match, otherwise False.
        """
        # Check all permutations of variables within each constraint
        for perm_constraint in perm_constraints:
            perm_operator = re.search(r'(==|<=|>=|=|<|>)', perm_constraint).group(0)
            for df_constraint in df_constraints:
                df_operator = re.search(r'(==|<=|>=|=|<|>)', df_constraint).group(0)
                lhs_perm, rhs_perm = perm_constraint.split(perm_operator)
                lhs_perm = re.split(r'(\+)', lhs_perm)
                lhs_df, rhs_df = df_constraint.split(df_operator)
                lhs_df = re.split(r'(\+)', lhs_df)
                if sorted(lhs_perm) == sorted(lhs_df) and rhs_perm == rhs_df and perm_operator == df_operator:
                    return True
        return False
