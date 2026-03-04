"""
run_knapsack.py -- Run QAOA experiments for knapsack constraints.

Reads data/knapsack_constraints.csv (format: n; ['a_0*x_0 + ... <= b']).

Usage:
    python run/run_knapsack.py --corp constraint --max_n 4
    python run/run_knapsack.py --corp problem --max_n 4 --n_layers 1
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


def run_cqaoa(max_n: int, result_dir: str = './results/',
              data_dir: str = './data/',
              result_file: str = 'knapsack_constraint_results',
              combined: bool = False, single_flag: bool = False,
              decompose: bool = True) -> None:
    """Run VCG on knapsack constraints."""
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'knapsack_constraints.csv')
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


def run_pqaoa(max_n: int, n_layers: int = 1,
              result_dir: str = './results/',
              data_dir: str = './data/',
              result_file: str = 'qubo_knapsack_results',
              constraint_result_file: str = 'knapsack_constraint_results',
              combined: bool = False, overlap: bool = False,
              single_flag: bool = False, decompose: bool = True) -> None:
    """Run ProblemQAOA on knapsack constraints paired with QUBOs."""
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'knapsack_constraints.csv')
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
    parser = argparse.ArgumentParser(description='Run QAOA for knapsack constraints.')
    parser.add_argument('--corp', type=str, default='constraint',
                        choices=['constraint', 'problem'],
                        help='Run constraint gadget training or QUBO problem solving')
    parser.add_argument('--max_n', type=int, default=4,
                        help='Maximum number of variables')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of QAOA layers (problem mode only)')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    if args.corp == 'constraint':
        run_cqaoa(args.max_n, result_dir=args.results_dir, data_dir=args.data_dir)
    elif args.corp == 'problem':
        run_pqaoa(args.max_n, n_layers=args.n_layers,
                  result_dir=args.results_dir, data_dir=args.data_dir)
    else:
        raise ValueError('--corp must be "constraint" or "problem"')
