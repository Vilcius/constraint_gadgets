"""
run_flow.py -- Run QAOA experiments for flow conservation constraints.

Reads data/flow_constraints.csv (format: n_vars; ['x_0 + ... - x_j - ... == 0']).

Usage:
    python run/run_flow.py --corp constraint --max_in 3 --max_out 3
    python run/run_flow.py --corp hybrid --n_layers 1
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


def run_vcg(max_in: int = 3, max_out: int = 3,
            result_dir: str = './results/',
            data_dir: str = './data/',
            result_file: str = 'flow_constraint_results') -> None:
    """Train VCG gadgets on all flow conservation constraints."""
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'flow_constraints.csv')
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path)
                       if n <= max_in + max_out]

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


def run_hybrid(max_in: int = 3, max_out: int = 3, n_layers: int = 1,
               result_dir: str = './results/',
               data_dir: str = './data/',
               result_file: str = 'hybrid_flow_results',
               constraint_result_file: str = 'flow_constraint_results') -> None:
    """Run HybridQAOA on flow constraints paired with QUBOs."""
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'flow_constraints.csv')
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path)
                       if n <= max_in + max_out]
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
    parser = argparse.ArgumentParser(description='Run QAOA for flow constraints.')
    parser.add_argument('--corp', type=str, default='constraint',
                        choices=['constraint', 'hybrid'],
                        help='Train VCG gadgets or run HybridQAOA')
    parser.add_argument('--max_in', type=int, default=3,
                        help='Maximum number of in-flow variables')
    parser.add_argument('--max_out', type=int, default=3,
                        help='Maximum number of out-flow variables')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of QAOA layers (hybrid mode only)')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    if args.corp == 'constraint':
        run_vcg(args.max_in, args.max_out,
                result_dir=args.results_dir, data_dir=args.data_dir)
    elif args.corp == 'hybrid':
        run_hybrid(args.max_in, args.max_out, n_layers=args.n_layers,
                   result_dir=args.results_dir, data_dir=args.data_dir)
    else:
        raise ValueError('--corp must be "constraint" or "hybrid"')
