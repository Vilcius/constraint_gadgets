"""
run_assignment.py -- Run QAOA experiments for assignment problem constraints.

Reads data/assignment_constraints.csv (format: n_vars; ['row_0', ..., 'col_0', ...]).

All assignment constraints (sum_j x_{i*n+j} == 1) are Dicke-compatible, so
VCG training produces no entries -- they are handled by exact DickeStatePrep
in HybridQAOA with a Ring-XY mixer.

Usage:
    python run/run_assignment.py --corp constraint --max_n 3
    python run/run_assignment.py --corp hybrid --max_n 3 --n_layers 1
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
from analyze_results.results_helper import (
    read_typed_csv, collect_vcg_data, collect_hybrid_data,
    remap_to_zero_indexed, remap_constraint_to_vars,
)


def run_vcg(max_n: int = 3, result_dir: str = './results/',
            data_dir: str = './data/',
            result_file: str = 'assignment_constraint_results') -> None:
    """Train VCGs for non-Dicke assignment constraints.

    All standard assignment constraints (sum x_{ij} == 1) are Dicke-compatible
    and are skipped here.  This function is kept for API consistency and in case
    future variants include non-Dicke constraints.
    """
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'assignment_constraints.csv')
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path)
                       if int(n ** 0.5) <= max_n]

    for n_vars, constraints in all_constraints:
        parsed = ch.parse_constraints(constraints)
        for pc in parsed:
            if ch.is_dicke_compatible(pc) or ch.is_flow_compatible(pc):
                continue  # Exact state prep; no VCG needed
            remapped, n_c_vars = remap_to_zero_indexed(pc.raw, pc.variables)
            flag_wire = n_c_vars
            for angsty in ['QAOA', 'ma-QAOA']:
                gadget = vcg_module.VCG(
                    constraints=[remapped],
                    flag_wires=[flag_wire],
                    angle_strategy=angsty,
                    n_layers=1,
                )
                row = collect_vcg_data(gadget)
                df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
                df.to_pickle(f'{result_dir}{result_file}.pkl')

    if df.empty:
        print("All assignment constraints are Dicke-compatible; no VCG training needed.")
    else:
        print(f"Saved {len(df)} rows to {result_dir}{result_file}.pkl")


def run_hybrid(max_n: int = 3, n_layers: int = 1,
               result_dir: str = './results/',
               data_dir: str = './data/',
               result_file: str = 'hybrid_assignment_results',
               constraint_result_file: str = 'assignment_constraint_results') -> None:
    """Run HybridQAOA on assignment constraints paired with QUBOs.

    All assignment constraints are Dicke-compatible, so HybridQAOA uses
    DickeStatePrep for each row/column constraint with a Ring-XY mixer.
    No flag qubits are needed; total qubits = n_vars only.
    """
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'assignment_constraints.csv')
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path)
                       if int(n ** 0.5) <= max_n]
    qubos = data.read_qubos_from_file('qubos.csv', results_dir=data_dir)
    gadget_path = f'{result_dir}{constraint_result_file}.pkl'

    for p in range(1, n_layers + 1):
        for n_vars, constraints in all_constraints:
            for n_qubo, qubo_dict in sorted(qubos.items()):
                if n_qubo < n_vars:
                    continue
                # Randomly assign constraint variables to a subset of QUBO positions.
                var_assignment = sorted(
                    np.random.choice(n_qubo, n_vars, replace=False).tolist()
                )
                remapped = [remap_constraint_to_vars(c, var_assignment) for c in constraints]
                parsed = ch.parse_constraints(remapped)
                structural_indices = list(range(len(parsed)))
                for q in qubo_dict:
                    min_val, optimal_x, total_min = data.get_optimal_x(
                        qubo_dict[q]['Q'], remapped
                    )
                    hybrid = hq.HybridQAOA(
                        qubo=qubo_dict[q]['Q'],
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
                            (df['qubo_string'] == qubo_dict[q]['qubo_string']) &
                            (df['constraints'].map(tuple) == tuple(remapped))
                        )
                        previous_angles = np.array(df[mask]['opt_angles'].values[0])
                    row = collect_hybrid_data(
                        remapped, hybrid, qubo_dict[q]['qubo_string'],
                        min_val=min_val, previous_angles=previous_angles,
                        var_assignment=var_assignment,
                    )
                    df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
                    df.to_pickle(f'{result_dir}{result_file}.pkl')

    print(f"Saved {len(df)} rows to {result_dir}{result_file}.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run QAOA for assignment constraints.')
    parser.add_argument('--corp', type=str, default='constraint',
                        choices=['constraint', 'hybrid'],
                        help='Train VCG gadgets or run HybridQAOA')
    parser.add_argument('--max_n', type=int, default=3,
                        help='Maximum assignment size (max_n × max_n problem)')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of QAOA layers (hybrid mode only)')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    if args.corp == 'constraint':
        run_vcg(args.max_n, result_dir=args.results_dir, data_dir=args.data_dir)
    elif args.corp == 'hybrid':
        run_hybrid(args.max_n, n_layers=args.n_layers,
                   result_dir=args.results_dir, data_dir=args.data_dir)
    else:
        raise ValueError('--corp must be "constraint" or "hybrid"')
