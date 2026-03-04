"""
run_quadratic.py -- Run QAOA experiments for quadratic constraints.

Reads:
    data/independent_set_constraints.csv      (--type independent_set)
    data/quadratic_knapsack_constraints.csv   (--type quadratic_knapsack)

Usage:
    python run/run_quadratic.py --corp constraint --type independent_set --max_n 5
    python run/run_quadratic.py --corp constraint --type quadratic_knapsack --max_n 4
    python run/run_quadratic.py --corp hybrid --type independent_set --max_n 5
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
from core import hybrid_qaoa as hq
from core import constraint_handler as ch
from data import make_data as data
from run.run_utils import read_typed_csv, collect_vcg_data, collect_hybrid_data

CSV_FILES = {
    'independent_set': 'independent_set_constraints.csv',
    'quadratic_knapsack': 'quadratic_knapsack_constraints.csv',
}


def run_vcg(constraint_type: str, max_n: int,
            result_dir: str = './results/',
            data_dir: str = './data/',
            result_file: str = None) -> None:
    """Train VCG gadgets on quadratic constraints up to max_n variables."""
    if result_file is None:
        result_file = f'{constraint_type}_constraint_results'
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, CSV_FILES[constraint_type])
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path) if n <= max_n]

    for n_vars, constraints in all_constraints:
        flag_wires = list(range(n_vars, n_vars + len(constraints)))
        for angsty in ['QAOA', 'ma-QAOA']:
            gadget = vcg_module.VCG(
                constraints=constraints,
                flag_wires=flag_wires,
                angle_strategy=angsty,
                n_layers=1,
            )
            row = collect_vcg_data(gadget)
            df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
            df.to_pickle(f'{result_dir}{result_file}.pkl')

    print(f"Saved {len(df)} rows to {result_dir}{result_file}.pkl")


def run_hybrid(constraint_type: str, max_n: int, n_layers: int = 1,
               result_dir: str = './results/',
               data_dir: str = './data/',
               result_file: str = None,
               constraint_result_file: str = None) -> None:
    """Run HybridQAOA on quadratic constraints paired with QUBOs."""
    if result_file is None:
        result_file = f'hybrid_{constraint_type}_results'
    if constraint_result_file is None:
        constraint_result_file = f'{constraint_type}_constraint_results'
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, CSV_FILES[constraint_type])
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path) if n <= max_n]
    qubos = data.read_qubos_from_file('qubos.csv', results_dir=data_dir)
    gadget_path = f'{result_dir}{constraint_result_file}.pkl'

    for p in range(1, n_layers + 1):
        for n_vars, constraints in all_constraints:
            if n_vars not in qubos:
                continue
            parsed = ch.parse_constraints(constraints)
            structural_indices = list(range(len(parsed)))
            for q in qubos[n_vars]:
                min_val, optimal_x, total_min = data.get_optimal_x(
                    qubos[n_vars][q]['Q'], constraints
                )
                hybrid = hq.HybridQAOA(
                    qubo=qubos[n_vars][q]['Q'],
                    all_constraints=parsed,
                    structural_indices=structural_indices,
                    penalty_indices=[],
                    penalty_str=[float(5 + 2 * np.abs(total_min))],
                    angle_strategy='ma-QAOA',
                    mixer='Grover',
                    n_layers=p,
                    learning_rate=0.01,
                    steps=100,
                    num_restarts=10,
                    pre_made=True,
                    gadget_path=gadget_path,
                )
                previous_angles = None
                if p > 1:
                    mask = (
                        (df['n_layers'] == p - 1) &
                        (df['qubo_string'] == qubos[n_vars][q]['qubo_string']) &
                        (df['constraints'].map(tuple) == tuple(constraints))
                    )
                    previous_angles = np.array(df[mask]['opt_angles'].values[0])
                row = collect_hybrid_data(
                    constraints, hybrid, qubos[n_vars][q]['qubo_string'],
                    min_val=min_val, previous_angles=previous_angles,
                )
                df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
                df.to_pickle(f'{result_dir}{result_file}.pkl')

    print(f"Saved {len(df)} rows to {result_dir}{result_file}.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run QAOA for quadratic constraints.')
    parser.add_argument('--corp', type=str, default='constraint',
                        choices=['constraint', 'hybrid'],
                        help='Train VCG gadgets or run HybridQAOA')
    parser.add_argument('--type', type=str, default='independent_set',
                        choices=['independent_set', 'quadratic_knapsack'],
                        help='Quadratic constraint type')
    parser.add_argument('--max_n', type=int, default=5,
                        help='Maximum number of variables')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of QAOA layers (hybrid mode only)')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    if args.corp == 'constraint':
        run_vcg(args.type, args.max_n,
                result_dir=args.results_dir, data_dir=args.data_dir)
    elif args.corp == 'hybrid':
        run_hybrid(args.type, args.max_n, n_layers=args.n_layers,
                   result_dir=args.results_dir, data_dir=args.data_dir)
    else:
        raise ValueError('--corp must be "constraint" or "hybrid"')
