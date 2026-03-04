"""
run_quadratic.py -- Run QAOA experiments for quadratic constraints.

Reads:
    data/independent_set_constraints.csv      (--type independent_set)
    data/quadratic_knapsack_constraints.csv   (--type quadratic_knapsack)

Usage:
    python run/run_quadratic.py --corp constraint --type independent_set --max_n 5
    python run/run_quadratic.py --corp constraint --type quadratic_knapsack --max_n 4
    python run/run_quadratic.py --corp problem --type independent_set
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pennylane import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

from core import vcg as vcg_module
from core import problem_qaoa as pq
from analyze_results import make_data as data
from run.run_utils import read_typed_csv, collect_vcg_data, collect_pqaoa_data

CSV_FILES = {
    'independent_set': 'independent_set_constraints.csv',
    'quadratic_knapsack': 'quadratic_knapsack_constraints.csv',
}


def run_vcg(constraint_type: str, max_n: int,
            result_dir: str = './results/',
            data_dir: str = './data/',
            result_file: str = None,
            combined: bool = False, single_flag: bool = False,
            decompose: bool = True) -> None:
    """Run VCG on quadratic constraints up to max_n variables."""
    if result_file is None:
        result_file = f'{constraint_type}_constraint_results'
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, CSV_FILES[constraint_type])
    all_constraints = read_typed_csv(csv_path)
    all_constraints = [(n, cs) for n, cs in all_constraints if n <= max_n]

    angle_strats = ['QAOA', 'ma-QAOA']
    for n_vars, constraints in all_constraints:
        flag_wires = list(range(n_vars, n_vars + len(constraints)))
        for angsty in angle_strats:
            gadget = vcg_module.VCG(
                constraints=constraints,
                flag_wires=flag_wires,
                angle_strategy=angsty,
                n_layers=1,
                pre_made=False,
                path=f'{result_dir}{result_file}.pkl',
            )
            row = collect_vcg_data(gadget, combined=combined,
                                   single_flag=single_flag, decompose=decompose)
            df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
            df.to_pickle(f'{result_dir}{result_file}.pkl')

    print(f"Saved {len(df)} rows to {result_dir}{result_file}.pkl")


def run_pqaoa(constraint_type: str, max_n: int, n_layers: int = 1,
              result_dir: str = './results/',
              data_dir: str = './data/',
              result_file: str = None,
              constraint_result_file: str = None,
              combined: bool = False, overlap: bool = False,
              single_flag: bool = False, decompose: bool = True) -> None:
    """Run ProblemQAOA on quadratic constraints paired with QUBOs."""
    if result_file is None:
        result_file = f'qubo_{constraint_type}_results'
    if constraint_result_file is None:
        constraint_result_file = f'{constraint_type}_constraint_results'
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, CSV_FILES[constraint_type])
    all_constraints = read_typed_csv(csv_path)
    all_constraints = [(n, cs) for n, cs in all_constraints if n <= max_n]

    qubos = data.read_qubos_from_file('qubos.csv', results_dir=data_dir)
    angle_strats = ['ma-QAOA']

    for p in range(1, n_layers + 1):
        for n_vars, constraints in all_constraints:
            if n_vars not in qubos:
                continue
            flag_wires = list(range(n_vars, n_vars + len(constraints)))
            for q in qubos[n_vars]:
                min_val, optimal_x, total_min = data.get_optimal_x(
                    qubos[n_vars][q]['Q'], constraints
                )
                for angsty in angle_strats:
                    cqaoa = vcg_module.VCG(
                        constraints=constraints,
                        flag_wires=flag_wires,
                        angle_strategy=angsty,
                        n_layers=1,
                        pre_made=True,
                        path=f'{result_dir}{constraint_result_file}.pkl',
                    )
                    pqaoa = pq.ProblemQAOA(
                        qubo=qubos[n_vars][q]['Q'],
                        state_prep=[cqaoa],
                        angle_strategy='ma-QAOA',
                        mixer='Grover',
                        penalty=[5 + 2 * np.abs(total_min)],
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
                        mask = (
                            (df['n_layers'] == p - 1) &
                            (df['qubo_string'] == qubos[n_vars][q]['qubo_string']) &
                            (df['constraints'].map(tuple) == tuple(constraints))
                        )
                        previous_angles = np.array(df[mask]['opt_angles'].values[0])
                    row = collect_pqaoa_data(
                        constraints, pqaoa, qubos[n_vars][q]['qubo_string'],
                        combined=combined, overlap=overlap,
                        single_flag=single_flag, decompose=decompose,
                        previous_angles=previous_angles, min_val=min_val,
                    )
                    df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
                    df.to_pickle(f'{result_dir}{result_file}.pkl')

    print(f"Saved {len(df)} rows to {result_dir}{result_file}.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run QAOA for quadratic constraints.')
    parser.add_argument('--corp', type=str, default='constraint',
                        choices=['constraint', 'problem'],
                        help='Run VCG training or QUBO problem solving')
    parser.add_argument('--type', type=str, default='independent_set',
                        choices=['independent_set', 'quadratic_knapsack'],
                        help='Quadratic constraint type')
    parser.add_argument('--max_n', type=int, default=5,
                        help='Maximum number of variables')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of QAOA layers (problem mode only)')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    if args.corp == 'constraint':
        run_vcg(args.type, args.max_n,
                result_dir=args.results_dir, data_dir=args.data_dir)
    elif args.corp == 'problem':
        run_pqaoa(args.type, args.max_n, n_layers=args.n_layers,
                  result_dir=args.results_dir, data_dir=args.data_dir)
    else:
        raise ValueError('--corp must be "constraint" or "problem"')
