import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import constraint_qaoa_server as cq
import problem_qaoa as pq
import make_data as data
import argparse
import warnings
warnings.filterwarnings('ignore')

#|%%--%%| <35nAGOsdcG|bERad0UHYE>
r"""°°°
Functions to collect data from ConstraintQAOA and ProblemQAOA instances after optimization.
°°°"""
#|%%--%%| <bERad0UHYE|XS2msgKae8>


def collect_cqaoa_data(cqaoa: cq.ConstraintQAOA, combined: bool = False, single_flag: bool = False, decompose: bool = True) -> dict:
    """Collect data from a ConstraintQAOA instance after optimization.
    Args:
        cqaoa (cq.ConstraintQAOA): An instance of the ConstraintQAOA class.
        combined (bool, optional): Whether constraints were combined. Defaults to False.
        single_flag (bool, optional): Whether a single flag qubit was used. Defaults to False.
        decompose (bool, optional): Whether the Hamiltonian was decomposed. Defaults to True.
    Returns:
        dict: A dictionary containing various metrics and results from the CQAOA instance.
    """
    C_max = max(qml.eigvals(cqaoa.constraint_Ham))
    C_min = min(qml.eigvals(cqaoa.constraint_Ham))
    opt_cost, opt_angles = cqaoa.optimize_angles(cqaoa.do_evolution_circuit)
    resources, est_shots, est_error, group_est_shots, group_est_error = cqaoa.get_circuit_resources()
    counts = cqaoa.do_counts_circuit(shots=est_shots)
    cqaoa_dataset = {
        'constraints': [cqaoa.constraints],
        'n_x': [cqaoa.n_x],
        'n_c': [len(cqaoa.constraints)],
        'combined': [combined],
        'single_flag': [single_flag],
        'decompose': [decompose],
        'Hamiltonian': [cqaoa.constraint_Ham],
        'outcomes': [cqaoa.outcomes],
        'angle_strategy': [cqaoa.angle_strategy],
        'n_layers': [cqaoa.n_layers],
        'num_gamma': [cqaoa.num_gamma],
        'num_beta': [cqaoa.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [opt_cost],
        'counts': [counts],
        'resources': [resources],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [cqaoa.hamiltonian_time],
        'optimize_time': [cqaoa.optimize_time],
        'counts_time': [cqaoa.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'AR': [(opt_cost-C_max)/(C_min-C_max)],
    }
    return cqaoa_dataset


def collect_pqaoa_data(constraints: list, pqaoa: pq.ProblemQAOA, qubo_string: str, combined=False, overlap=False, single_flag=False, decompose=True, previous_angles=None, min_val=None) -> dict:
    """
    Collect data from a ProblemQAOA instance after optimization.
    Args:
        constraints (list): List of constraint strings used in the problem.
        pqaoa (pq.ProblemQAOA): An instance of the ProblemQAOA class.
        qubo_string (str): String representation of the QUBO used in the problem.
        combined (bool, optional): Whether constraints were combined. Defaults to False.
        overlap (bool, optional): Whether overlapping variables were used. Defaults to False.
        single_flag (bool, optional): Whether a single flag qubit was used. Defaults to False.
        decompose (bool, optional): Whether the Hamiltonian was decomposed. Defaults to True.
        previous_angles (np.ndarray, optional): Optimal angles from the previous layer for warm-starting. Defaults to None.
        min_val (float, optional): Minimum value of the objective function for feasible solutions. Defaults to None.
    Returns:
        dict: A dictionary containing various metrics and results from the PQAOA instance.
    """
    C_max = max(qml.eigvals(pqaoa.problem_Ham))
    C_min = min(qml.eigvals(pqaoa.problem_Ham))
    if pqaoa.n_layers == 1:
        opt_cost, opt_angles = pqaoa.optimize_angles(pqaoa.do_evolution_circuit)
    else:
        opt_cost, opt_angles = pqaoa.optimize_angles(pqaoa.do_evolution_circuit, prev_layer_angles=previous_angles)
    resources, est_shots, est_error, group_est_shots, group_est_error = pqaoa.get_circuit_resources()
    counts = pqaoa.do_counts_circuit(shots=10000)
    pqaoa_dataset = {
        'qubo_string': [qubo_string],
        'constraints': [constraints],
        'n_x': [pqaoa.n_x],
        'n_c': [constraints],
        'combined': [combined],
        'constraint_penalty': [pqaoa.penalty],
        'overlap': [overlap],
        'overlap_vars': [pqaoa.overlap_vars],
        'overlap_penalty': [pqaoa.overlap_penalty],
        'single_flag': [single_flag],
        'decompose': [decompose],
        'Hamiltonian': [pqaoa.problem_Ham],
        'outcomes': [None],
        'angle_strategy': [pqaoa.angle_strategy],
        'n_layers': [pqaoa.n_layers],
        'num_gamma': [pqaoa.num_gamma],
        'num_beta': [pqaoa.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [opt_cost],
        'counts': [counts],
        'resources': [resources],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [pqaoa.hamiltonian_time],
        'optimize_time': [pqaoa.optimize_time],
        'counts_time': [pqaoa.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'min_val': [min_val],
        'AR': [(opt_cost-C_max)/(min_val-C_max)],
        'optimal_x': [pqaoa.optimal_x],
    }
    return pqaoa_dataset


#|%%--%%| <XS2msgKae8|uK2BmHTCtH>
r"""°°°
Functions to run ConstraintQAOA and ProblemQAOA for a range of constraints and collect results.
°°°"""
#|%%--%%| <uK2BmHTCtH|ZROy2CM4Gt>


def run_cqaoa(max_n: int, op: str, result_dir='./results/', result_file='single_constraint', combined=False, single_flag=False, decompose=True) -> None:
    r"""
    Run ConstraintQAOA for constraints of the form $\sum_{i=1}^n x_i \; op \; b$ for n in [1, max_n] and b in [0, n].
    Then writes results to pickle file.
    Args:
        max_n (int): Maximum number of variables in the constraint.
        op (str): Operator for the constraint ('==', '>=', '<=', '<', '>').
        result_dir (str, optional): Directory to save results. Defaults to './results/'.
        result_file (str, optional): Filename for saving results. Defaults to 'single_constraint'.
        combined (bool, optional): Whether to combine constraints. Defaults to False.
        single_flag (bool, optional): Whether to use a single flag qubit. Defaults to False.
        decompose (bool, optional): Whether to decompose the Hamiltonian. Defaults to True.
    Returns:
        None
    """
    df = pd.DataFrame(columns=['constraints', 'n_x', 'n_c', 'single_flag', 'decompose', 'outcomes', 'Hamiltonian', 'angle_strategy', 'n_layers', 'num_gamma', 'num_beta', 'opt_angles', 'opt_cost', 'counts', 'resources', 'est_shots', 'est_error', 'group_est_shots', 'group_est_error', 'optimize_time', 'table_time', 'hamiltonian_time', 'counts_time'])
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    angle_strats = ['QAOA', 'ma-QAOA']
    for n in range(1, max_n+1):
        for b in range(n+1):
            constraint = data.make_single_constraint(n, op, b)
            for angsty in angle_strats:
                cqaoa = cq.ConstraintQAOA(
                    constraints=[constraint],
                    flag_wires=[n],
                    angle_strategy=angsty,
                    n_layers=1,
                    pre_made=False,
                    path=f'{result_dir}{result_file}.pkl',
                )

                cqaoa_dataset = collect_cqaoa_data(cqaoa, combined=combined, single_flag=single_flag, decompose=decompose)
                df = pd.concat([df, pd.DataFrame(cqaoa_dataset)], ignore_index=True)
                df.to_pickle(f'{result_dir}{result_file}.pkl')


def run_pqaoa(max_n: int, op: str, result_dir='./results/', result_file='qubo_single_constraint', n_layers=1, combined=False, overlap=False, single_flag=False, decompose=True, constraint_result_file='single_constraint') -> None:
    r"""
    Run ProblemQAOA for QUBOs with constraints of the form $\sum_{i=1}^n x_i \; op \; b$ for n in [1, max_n] and b in [0, n].
    Then writes results to pickle file.
    Args:
        max_n (int): Maximum number of variables in the constraint.
        op (str): Operator for the constraint ('==', '>=', '<=', '<', '>').
        result_dir (str, optional): Directory to save results. Defaults to './results/'.
        result_file (str, optional): Filename for saving results. Defaults to 'qubo_single_constraint'.
        n_layers (int, optional): Number of QAOA layers. Defaults to 1.
        combined (bool, optional): Whether to combine constraints. Defaults to False.
        overlap (bool, optional): Whether to use overlapping variables. Defaults to False.
        single_flag (bool, optional): Whether to use a single flag qubit. Defaults to False.
        decompose (bool, optional): Whether to decompose the Hamiltonian. Defaults to True.
        constraint_result_file (str, optional): Filename of the constraint results to use. Defaults to 'single_constraint'.
    Returns:
        None
    """
    df = pd.DataFrame(columns=['qubo_string', 'constraints', 'n_x', 'n_c', 'combined', 'constraint_penalty', 'overlap', 'overlap_vars', 'overlap_penalty', 'single_flag', 'decompose', 'outcomes', 'Hamiltonian', 'angle_strategy', 'n_layers', 'num_gamma', 'num_beta', 'opt_angles', 'opt_cost', 'counts', 'resources', 'est_shots', 'est_error', 'group_est_shots', 'group_est_error', 'optimize_time', 'hamiltonian_time', 'counts_time', 'C_max', 'C_min', 'AR'])
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    qubos = data.read_qubos_from_file('qubos.csv', results_dir=result_dir)

    qubo_opt_x = {}
    angle_strats = ['ma-QAOA']
    for p in range(1, n_layers+1):
        for n in range(1, max_n+1):
            for b in range(n+1):
                constraint = data.make_single_constraint(n, op, b)
                for q in qubos[n]:
                    min_val, optimal_x, total_min = data.get_optimal_x(qubos[n][q]['Q'], [constraint])
                    qubo_opt_x[(qubos[n][q]['qubo_string'], str(constraint))] = optimal_x
                    for angsty in angle_strats:
                        cqaoa = cq.ConstraintQAOA(
                            constraints=[constraint],
                            flag_wires=[n],
                            angle_strategy=angsty,
                            n_layers=1,
                            pre_made=True,
                            path=f'{result_dir}{constraint_result_file}.pkl',
                        )
                        pqaoa = pq.ProblemQAOA(
                            qubo=qubos[n][q]['Q'],
                            state_prep=[cqaoa],  # [cqaoa],
                            angle_strategy="ma-QAOA",
                            mixer="Grover",
                            penalty=[5+2 * np.abs(total_min)],
                            n_layers=p,
                            samples=10000,
                            learning_rate=0.01,
                            steps=100,
                            num_restarts=10,
                        )
                        pqaoa.optimal_x = optimal_x
                        if p == 1:
                            previous_angles = None
                        else:
                            previous_angles = np.array(df[(df['n_layers'] == p-1) & (df['qubo_string'] == qubos[n][q]['qubo_string']) & (df['constraints'].map(tuple) == tuple(constraint))]['opt_angles'].values[0])

                        pqaoa_dataset = collect_pqaoa_data(constraint, pqaoa, qubos[n][q]['qubo_string'], combined=False, overlap=False, single_flag=single_flag, decompose=decompose, previous_angles=previous_angles, min_val=min_val)
                        df = pd.concat([df, pd.DataFrame(pqaoa_dataset)], ignore_index=True)
                        df.to_pickle(f'{result_dir}{result_file}.pkl')


#|%%--%%| <ZROy2CM4Gt|IQyqOAodvj>
r"""°°°
Run either the constraint gadget creation or the QCBO problem solving based on command line arguments.
°°°"""
#|%%--%%| <IQyqOAodvj|31MMWsyyX4>


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the server code.')
    parser.add_argument('--max_n', type=int, default=5, help='Maximum number of variables')
    parser.add_argument('--op', type=str, default='equals', help='Operator for the constraint')
    parser.add_argument('--results_dir', type=str, default='./results/', help='Directory to save results')
    parser.add_argument('--corp', type=str, default='constraint', help='Whether to make the constraint gadget or solve the qcbo problem')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers for QAOA')
    args = parser.parse_args()
    ops = {'equals': '==', 'geq': '>=', 'leq': '<=', 'less': '<', 'greater': '>'}

    if args.corp == 'constraint':
        run_cqaoa(args.max_n, ops[args.op], result_dir='./results/', result_file=f'single_constraint_{args.op}')
    elif args.corp == 'problem':
        run_pqaoa(args.max_n, ops[args.op], result_dir='./results/', result_file=f'qubo_single_constraint_{args.op}_{args.n_layers}', n_layers=args.n_layers, combined=False, overlap=False, single_flag=False, decompose=True, constraint_result_file=f'single_constraint_{args.op}')
    else:
        raise ValueError('Invalid value for --corp. Must be "constraint" or "problem".')


