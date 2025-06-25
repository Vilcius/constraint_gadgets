from pennylane import numpy as np
import itertools as it
import os
import warnings
warnings.filterwarnings('ignore')


#|%%--%%| <r7FwNHU1N7|c49bRzGd27>
r"""°°°
Functions to create single linear constraints.
°°°"""
#|%%--%%| <c49bRzGd27|FGDZWJe8Bm>


def make_single_constraint(n: int, op: str, b: int, coeff: list = None) -> str:
    """
    Create a linear constraint of the form $\\sum_{i=1}^n a_i x_i op b$.
    Args:
        n (int): Number of variables.
        op (str): Operator for the constraint ('==', '>=', '<=', '<', '>').
        b (int): Right-hand side value of the constraint.
        coeff (list, optional): Coefficients for each variable. Defaults to None, which sets all coefficients to 1.
    Returns:
        str: A string representing the linear constraint.
    """
    if coeff is None:
        coeff = [1] * n
    # Create the constraint string
    constraint = " + ".join([f"{coeff[i]}*x_{i}" for i in range(n)]) + f" {op} {b}"
    return constraint


def make_two_constraint(support: int) -> list[str]:
    """
    Generate a random constraint with the specified support.
    Args:
        support (int): The support of the constraint (1, 2, or 3).
    Returns:
        list[str]: A list of two constraint strings.
    """
    lhs = {
        1: ['x_0 + x_1 + x_2', 'x_0 + x_3 + x_4'],
        2: ['x_0 + x_1 + x_2 + x_3', 'x_0 + x_3 + x_4'],
        3: ['x_0 + x_1 + x_2 + x_3', 'x_0 + x_2 + x_3 + x_4'],
    }

    ops = ['==', '<=', '>=']

    rhs = {1: range(4), 2: range(5), 3: range(5)}[support]
    lhss = lhs[support]
    opss = np.random.choice(ops, size=2, replace=True)
    rhss = np.random.choice(rhs, size=2, replace=True)
    constraints = [f"{lhss[i]} {opss[i]} {rhss[i]}" for i in range(2)]
    return constraints


def make_two_constraints(result_dir: str = './results/') -> None:
    """
    Generate constraints with specified support and at least one feasible solution.
    Then save them to a file in the specified directory.
    Args:
        result_dir (str): Directory to save the constraints file.
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # only generate constraints if the file doesn't already exist
    if not os.path.exists(f'{result_dir}two_constraints_support_1.txt'):
        for support in range(1, 4):
            constraints = {}
            num_constraints = 10
            feasible = False
            print('-'*40)
            print(f"Generating {num_constraints} constraints with support {support}")
            for i in range(num_constraints):
                print(f"Constraint {i+1}/{num_constraints}")
                while not feasible:
                    constraint = make_two_constraint(support)
                    print(constraint)
                    if constraint in constraints.values():
                        print("Duplicate constraint, regenerating...")
                        continue
                    feasible = check_feasibility(constraint)
                    print("Feasible:", feasible)
                constraints[i] = constraint
                feasible = False
                with open(f'{result_dir}two_constraints_support_{support}.txt', 'a') as f:
                    f.write(f"{constraint}\n")


def check_feasibility(constraints: list[str]) -> bool:
    """
    Check if there exists a feasible solution for the given constraints.
    Args:
        constraints (list[str]): A list of constraint strings.
    Returns:
        bool: True if there exists a feasible solution, False otherwise.
    """
    n_vars = 5
    all_bitstrings = it.product([0, 1], repeat=n_vars)
    for x in all_bitstrings:
        x_feasible = True
        var_dict = {f'x_{i}': x[i] for i in range(len(x))}
        for c in constraints:
            if not eval(c, var_dict):
                x_feasible = False
                break
        if x_feasible:
            return True
    return False


#|%%--%%| <FGDZWJe8Bm|aEkoiyklK3>
r"""°°°
Functions to create random QUBOs and read/write them to/from files.
°°°"""
#|%%--%%| <aEkoiyklK3|EG86IUpVs9>


def generate_random_qubo_string(n: int) -> [np.ndarray, str]:
    """
    Generate a random QUBO matrix and its corresponding string representation.
    Args:
        n (int): Size of the QUBO matrix (n x n).
    Returns:
        tuple:
            - np.ndarray: The generated QUBO matrix.
            - str: The string representation of the QUBO.
    """
    Q = np.random.randint(low=-5, high=5, size=(n, n))
    Q = np.triu(Q)  # Make the matrix upper triangular
    qubo_terms = []
    for i in range(len(Q)):
        for j in range(i, len(Q)):
            if Q[i, j] != 0:
                if i == j:
                    qubo_terms.append(f"{Q[i, j]}*x_{i}")
                else:
                    qubo_terms.append(f"{Q[i, j]}*x_{i} x_{j}")

    qubo_string = " + ".join(qubo_terms)
    return Q, qubo_string


def generate_n_qubos(num_qubo: int, min_n: int, max_n: int) -> dict:
    """
    Generate a dictionary of random QUBO matrices and their string representations for sizes ranging from min_n to max_n.
    Args:
        num_qubo (int): Number of QUBO matrices to generate for each size.
        min_n (int): Minimum size of the QUBO matrix.
        max_n (int): Maximum size of the QUBO matrix.
    Returns:
        dict: A dictionary where keys are sizes (n) and values are dictionaries containing QUBO matrices and their string representations.
    """
    qubos = {}
    for i in range(min_n, max_n+1):
        qubos[i] = {}
        for j in range(num_qubo):
            Q, qubo_string = generate_random_qubo_string(i)
            qubos[i][j] = {'Q': Q, 'qubo_string': qubo_string}
    return qubos


def write_qubos_to_file(qubos: dict, file_name: str, min_n: int, max_n: int, results_dir: str = './results/') -> None:
    """
    Write a dictionary of QUBO matrices and their string representations to a file.
    Args:
        qubos (dict): A dictionary where keys are sizes (n) and values are dictionaries containing QUBO matrices and their string representations.
        file_name (str): Name of the file to write the QUBOs to.
        min_n (int): Minimum size of the QUBO matrix.
        max_n (int): Maximum size of the QUBO matrix.
        results_dir (str, optional): Directory to save the file. Defaults to './results/'.
    """
    with open(f'{results_dir}{file_name}', 'w') as f:
        for i in range(min_n, max_n+1):
            for j in range(len(qubos[i])):
                f.write(f"({i}, {j}); {qubos[i][j]['qubo_string']}; {qubos[i][j]['Q'].tolist()}\n")


def read_qubos_from_file(file_name: str, results_dir: str = './results/') -> dict:
    """
    Read a file containing QUBO matrices and their string representations into a dictionary.
    Args:
        file_name (str): Name of the file to read the QUBOs from.
        results_dir (str, optional): Directory where the file is located. Defaults to './results/'.
    Returns:
        dict: A dictionary where keys are sizes (n) and values are dictionaries containing QUBO matrices and their string representations.
    """
    qubos = {}
    with open(f'{results_dir}{file_name}', 'r') as f:
        qubos = {}
        for i, line in enumerate(f):
            line = line.split(';')
            n_i = eval(line[0])
            n = n_i[0]
            if n not in qubos:
                qubos[n] = {}
            qubos[n][n_i[1]] = {}
            qubos[n][n_i[1]]['qubo_string'] = line[1]
            qubos[n][n_i[1]]['Q'] = np.array(eval(line[2]))
    return qubos


def get_optimal_x(qubo: np.ndarray, constraints: list) -> [float, list, float]:
    """
    Get the optimal solution for a given QUBO matrix and a list of constraints by brute force.
    Args:
        qubo (np.ndarray): The QUBO matrix.
        constraints (list): A list of constraint strings.
    Returns:
        tuple:
            - float: Minimum value of the objective function for feasible solutions.
            - list: List of optimal bitstrings (as strings) that achieve the minimum value.
            - float: Minimum value of the objective function across all solutions (feasible and infeasible).
    """
    all_bitstrings = it.product([0, 1], repeat=len(qubo))
    feasible = []
    infeasible = []
    for x in all_bitstrings:
        x_feasible = True
        var_dict = {f'x_{i}': x[i] for i in range(len(x))}
        for c in constraints:
            if not eval(c, var_dict):
                x_feasible = False
                break
        if x_feasible:
            feasible.append(x)
        else:
            infeasible.append(x)

    feas_vals = {}
    infeas_vals = {}

    # Calculate objective values for feasible solutions
    for x in feasible:
        feas_vals[''.join(map(str, x))] = np.dot(x, np.dot(qubo, x))

    # Calculate objective values for infeasible solutions
    for x in infeasible:
        infeas_vals[''.join(map(str, x))] = np.dot(x, np.dot(qubo, x))

    # Determine minimum values and optimal solutions if there are no feasible solutions
    if len(feas_vals) == 0:
        min_infeas_val = min(infeas_vals.values())
        return min_infeas_val, None, min_infeas_val

    # Determine minimum values and optimal solutions if there are feasible solutions
    if len(infeas_vals) == 0:
        min_feas_val = min(feas_vals.values())
        optimal_x = [k for k, v in feas_vals.items() if v == min_feas_val]
        return min_feas_val, optimal_x, min_feas_val

    # Determine minimum values and optimal solutions if there are both feasible and infeasible solutions
    min_feas_val = min(feas_vals.values())
    min_infeas_val = min(infeas_vals.values())
    optimal_x = [k for k, v in feas_vals.items() if v == min_feas_val]
    total_min_val = min(min_feas_val, min_infeas_val)

    return min_feas_val, optimal_x, total_min_val

