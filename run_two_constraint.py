import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import constraint_qaoa_server as cq
import problem_qaoa as pq
import make_data as data
import argparse
import warnings
warnings.filterwarnings('ignore')


#|%%--%%| <wSV4bXABkH|AMcG012Cru>
r"""°°°
Functions to collect data from ConstraintQAOA and ProblemQAOA instances after optimization.
°°°"""
#|%%--%%| <AMcG012Cru|vvC4czw1MY>


def collect_cqaoa_data(cqaoa: cq.ConstraintQAOA, combined: bool = False, single_flag: bool = False, decompose: bool = True) -> dict:
    """
    Collect data from a ConstraintQAOA instance after optimization.
    Args:
        cqaoa (cq.ConstraintQAOA): An instance of the ConstraintQAOA class.
        combined (bool, optional): Whether to use combined constraints. Defaults to False.
        single_flag (bool, optional): Whether to use a single flag qubit. Defaults to False.
        decompose (bool, optional): Whether to decompose multi-qubit gates. Defaults to True.
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
        'n_c': [2],
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


def collect_pqaoa_data(pqaoa: pq.ProblemQAOA, qubo_string: str, cqaoa: cq.ConstraintQAOA, min_val: float, combined: bool = False, overlap: bool = False, single_flag: bool = False, decompose: bool = True, support: int = 1) -> dict:
    """
    Collect data from a ProblemQAOA instance after optimization.
    Args:
        pqaoa (pq.ProblemQAOA): An instance of the ProblemQAOA class.
        qubo_string (str): String representation of the QUBO matrix.
        cqaoa (cq.ConstraintQAOA): The corresponding ConstraintQAOA instance used for state preparation.
        min_val (float): Minimum value of the objective function for feasible solutions.
        combined (bool, optional): Whether to use combined constraints. Defaults to False.
        overlap (bool, optional): Whether to use overlapping variables in constraints. Defaults to False.
        single_flag (bool, optional): Whether to use a single flag qubit. Defaults to False.
        decompose (bool, optional): Whether to decompose multi-qubit gates. Defaults to True.
        support (int, optional): Support of the constraints (1, 2, or 3). Defaults to 1.
    Returns:
        dict: A dictionary containing various metrics and results from the PQAOA instance.
    """
    C_max = max(qml.eigvals(pqaoa.problem_Ham))
    C_min = min(qml.eigvals(pqaoa.problem_Ham))

    if pqaoa.n_layers == 1:
        opt_cost, opt_angles = pqaoa.optimize_angles(pqaoa.do_evolution_circuit)
    elif pqaoa.n_layers == 2:
        previous_df = pd.read_pickle(f'./results/qubo_two_constraint_support_{support}_single_{single_flag}.pkl')
        previous_angles = np.array(previous_df[(previous_df['qubo_string'] == qubo_string) & (previous_df['constraints'].map(tuple) == tuple(cqaoa.constraints)) & (previous_df['angle_strategy'] == cqaoa.angle_strategy)]['opt_angles'].values[0])
        opt_cost, opt_angles = pqaoa.optimize_angles(pqaoa.do_evolution_circuit, prev_layer_angles=previous_angles)
    else:
        previous_df = pd.read_pickle(f'./results/qubo_two_constraint_support_{support}_single_{single_flag}_n_layers_{pqaoa.n_layers-1}.pkl')
        previous_angles = np.array(previous_df[(previous_df['qubo_string'] == qubo_string) & (previous_df['constraints'].map(tuple) == tuple(cqaoa.constraints)) & (previous_df['angle_strategy'] == cqaoa.angle_strategy)]['opt_angles'].values[0])
        opt_cost, opt_angles = pqaoa.optimize_angles(pqaoa.do_evolution_circuit, prev_layer_angles=previous_angles)

    opt_cost, opt_angles = pqaoa.optimize_angles(pqaoa.do_evolution_circuit)
    resources, est_shots, est_error, group_est_shots, group_est_error = pqaoa.get_circuit_resources()
    counts = pqaoa.do_counts_circuit(shots=10000)
    pqaoa_dataset = {
        'qubo_string': [qubo_string],
        'constraints': [cqaoa.constraints],
        'n_x': [pqaoa.n_x],
        'n_c': [2],
        'combined': [combined],
        'constraint_penalty': [pqaoa.penalty],
        'overlap': [overlap],
        'overlap_vars': [pqaoa.overlap_vars],
        'overlap_penalty': [pqaoa.overlap_penalty],
        'single_flag': [single_flag],
        'decompose': [decompose],
        'Hamiltonian': [pqaoa.problem_Ham],
        'outcomes': [cqaoa.outcomes],
        'angle_strategy': [cqaoa.angle_strategy],
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
        'AR': [(opt_cost-C_max)/(np.float64(min_val)-np.float64(C_max))],
        'optimal_x': [pqaoa.optimal_x],
    }
    return pqaoa_dataset


#|%%--%%| <vvC4czw1MY|QzKeZMU5aC>
r"""°°°
Functions to run ConstraintQAOA and ProblemQAOA for a range of constraints and collect results.
°°°"""
#|%%--%%| <QzKeZMU5aC|WNJGvQ0iTO>


def run_cqaoa(support: int, result_dir: str = './results/', result_file: str = 'two_constraint', combined: bool = False, single_flag: bool = False, decompose: bool = True) -> None:
    """
    Run ConstraintQAOA for a set of constraints and collect results.
    Args:
        support (int): Support of the constraints (1, 2, or 3).
        result_dir (str, optional): Directory to save results. Defaults to './results/'.
        result_file (str, optional): Filename to save results. Defaults to 'two_constraint'.
        combined (bool, optional): Whether to use combined constraints. Defaults to False.
        single_flag (bool, optional): Whether to use a single flag qubit. Defaults to False.
        decompose (bool, optional): Whether to decompose multi-qubit gates. Defaults to True.
    Returns:
        None
    """
    df = pd.DataFrame(columns=['constraints', 'n_x', 'n_c', 'single_flag', 'decompose', 'outcomes', 'Hamiltonian', 'angle_strategy', 'n_layers', 'num_gamma', 'num_beta', 'opt_angles', 'opt_cost', 'counts', 'resources', 'est_shots', 'est_error', 'group_est_shots', 'group_est_error', 'optimize_time', 'table_time', 'hamiltonian_time', 'counts_time'])
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    if single_flag:
        flag_wires = [5]
    else:
        flag_wires = [5, 6]

    with open(f'{result_dir}two_constraints_support_{support}.txt', 'r') as f:
        constraints_list = f.readlines()
    angle_strats = ['QAOA', 'ma-QAOA']
    for constraint in constraints_list:
        for angsty in angle_strats:
            cqaoa = cq.ConstraintQAOA(
                constraints=eval(constraint),
                flag_wires=flag_wires,
                angle_strategy=angsty,
                n_layers=1,
                pre_made=False,
                single_flag=single_flag,
                path=f'{result_dir}{result_file}.pkl',
            )

            cqaoa_dataset = collect_cqaoa_data(cqaoa, combined=combined, single_flag=single_flag, decompose=decompose)
            df = pd.concat([df, pd.DataFrame(cqaoa_dataset)], ignore_index=True)
            df.to_pickle(f'{result_dir}{result_file}.pkl')


def run_pqaoa(support: int, result_dir: str = './results/', constraint_result_file: str = 'two_constraint', result_file: str = 'qubo_two_constraint', combined: bool = False, overlap: bool = False, single_flag: bool = False, decompose: bool = True, n_layers: int = 1) -> None:
    """
    Run ProblemQAOA for a set of QCBOs and constraints, using ConstraintQAOA for state preparation, and collect results.
    Args:
        support (int): Support of the constraints (1, 2, or 3).
        result_dir (str, optional): Directory to save results. Defaults to './results/'.
        constraint_result_file (str, optional): Filename of the ConstraintQAOA results. Defaults to 'two_constraint'.
        result_file (str, optional): Filename to save ProblemQAOA results. Defaults to 'qubo_two_constraint'.
        combined (bool, optional): Whether to use combined constraints. Defaults to False.
        overlap (bool, optional): Whether to use overlapping variables in constraints. Defaults to False.
        single_flag (bool, optional): Whether to use a single flag qubit. Defaults to False.
        decompose (bool, optional): Whether to decompose multi-qubit gates. Defaults to True.
        n_layers (int, optional): Number of layers for QAOA. Defaults to 1.
    Returns:
        None
    """
    df = pd.DataFrame(columns=['qubo_string', 'constraints', 'n_x', 'n_c', 'combined', 'constraint_penalty', 'overlap', 'overlap_vars', 'overlap_penalty', 'single_flag', 'decompose', 'outcomes', 'Hamiltonian', 'angle_strategy', 'n_layers', 'num_gamma', 'num_beta', 'opt_angles', 'opt_cost', 'counts', 'resources', 'est_shots', 'est_error', 'group_est_shots', 'group_est_error', 'optimize_time', 'hamiltonian_time', 'counts_time', 'C_max', 'C_min', 'AR'])
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    qubos = data.read_qubos_from_file('qubos.csv', results_dir=result_dir)
    with open(f'{result_dir}two_constraints_support_{support}.txt', 'r') as f:
        constraints_list = f.readlines()

    if single_flag:
        flag_wires = [5]
    else:
        flag_wires = [5, 6]
    p = n_layers
    n = 5
    qubo_opt_x = {}
    angle_strats = ['ma-QAOA']
    for constraint in constraints_list:
        for q in qubos[n]:
            min_val, optimal_x, total_min = data.get_optimal_x(qubos[n][q]['Q'], eval(constraint))
            qubo_opt_x[(qubos[n][q]['qubo_string'], str(constraint))] = optimal_x
            for angsty in angle_strats:
                cqaoa = cq.ConstraintQAOA(
                    constraints=eval(constraint),
                    flag_wires=flag_wires,
                    angle_strategy=angsty,
                    n_layers=1,
                    single_flag=single_flag,
                    pre_made=True,
                    path=f'{result_dir}{constraint_result_file}.pkl',
                )
                pqaoa = pq.ProblemQAOA(
                    qubo=np.array(qubos[n][q]['Q'], requires_grad=False),
                    state_prep=[cqaoa],
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

                pqaoa_dataset = collect_pqaoa_data(pqaoa, qubos[n][q]['qubo_string'], cqaoa, min_val, combined=combined, overlap=overlap, single_flag=single_flag, decompose=decompose, support=support)
                df = pd.concat([df, pd.DataFrame(pqaoa_dataset)], ignore_index=True)
                df.to_pickle(f'{result_dir}{result_file}.pkl')


#|%%--%%| <WNJGvQ0iTO|j9rbWJUvDw>
r"""°°°
Main function to run either CQAOA or PQAOA based on command line arguments.
°°°"""
#|%%--%%| <j9rbWJUvDw|N2OB3lvR24>

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the server code.')
    parser.add_argument('--results_dir', type=str, default='./results/', help='Directory to save results')
    parser.add_argument('--corp', type=str, default='constraint', help='Whether to make the constraint gadget or solve the qubo problem')
    parser.add_argument('--support', type=int, default=1, help='Support of the constraints (1, 2, or 3)')
    parser.add_argument('--single_flag', action='store_true', help='Use single flag qubit for constraints')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers for QAOA')
    args = parser.parse_args()

    if args.corp == 'constraint':
        run_cqaoa(args.support, result_dir='./results/', result_file=f'two_constraint_support_{args.support}_single_{args.single_flag}', combined=False, single_flag=args.single_flag, decompose=True)
    elif args.corp == 'problem':
        run_pqaoa(args.support, result_dir='./results/', constraint_result_file=f'two_constraint_support_{args.support}_single_{args.single_flag}', result_file=f'qubo_two_constraint_support_{args.support}_single_{args.single_flag}_n_layers_{args.n_layers}', combined=False, overlap=False, single_flag=args.single_flag, decompose=True, n_layers=args.n_layers)
    else:
        raise ValueError('Invalid value for --corp. Must be "constraint" or "problem".')


