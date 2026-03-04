"""
run_subtour.py -- Run QAOA experiments for TSP subtour elimination constraints.

Reads data/subtour_constraints.csv (format: n_vars; ['c1', 'c2', ...]).
Each row contains the full set of subtour + assignment constraints for k cities,
where n_vars = k^2.

Dicke-compatible constraints (assignment rows: sum x_i == 1) are skipped during
VCG training -- they are handled by exact DickeStatePrep in HybridQAOA.
Non-Dicke constraints (subtour inequalities) are trained as individual VCGs
with variables remapped to 0-indexed.

Usage:
    python run/run_subtour.py --corp constraint --max_cities 3
    python run/run_subtour.py --corp hybrid --n_layers 1
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
from analyze_results import make_data as data
from run.run_utils import (
    read_typed_csv, collect_vcg_data, collect_hybrid_data,
    remap_to_zero_indexed,
)


def run_vcg(max_cities: int = 3,
            result_dir: str = './results/',
            data_dir: str = './data/',
            result_file: str = 'subtour_constraint_results') -> None:
    """Train individual VCGs for non-Dicke subtour inequality constraints.

    Dicke-compatible constraints (assignment rows) are skipped -- they need
    no VCG training.  Each subtour inequality constraint is trained separately
    with variables remapped to 0-indexed so that the saved Hamiltonian uses
    canonical wire labels that HybridQAOA can look up via ConstraintMapper.
    """
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'subtour_constraints.csv')
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path)
                       if int(n ** 0.5 + 0.5) <= max_cities]

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
        print("All subtour constraints are Dicke/Flow-compatible; no VCG training needed.")
    else:
        print(f"Saved {len(df)} rows to {result_dir}{result_file}.pkl")


def run_hybrid(max_cities: int = 3, n_layers: int = 1,
               result_dir: str = './results/',
               data_dir: str = './data/',
               result_file: str = 'hybrid_subtour_results',
               constraint_result_file: str = 'subtour_constraint_results') -> None:
    """Run HybridQAOA on subtour constraints paired with QUBOs.

    HybridQAOA automatically routes:
      - Dicke constraints (assignment rows) -> DickeStatePrep (no flags)
      - Subtour inequality constraints      -> individual VCGs (1 flag each,
                                               pre-loaded from gadget_path)
    """
    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame()
    df.to_pickle(f'{result_dir}{result_file}.pkl')

    csv_path = os.path.join(data_dir, 'subtour_constraints.csv')
    all_constraints = [(n, cs) for n, cs in read_typed_csv(csv_path)
                       if int(n ** 0.5 + 0.5) <= max_cities]
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
    parser = argparse.ArgumentParser(
        description='Run QAOA for TSP subtour elimination constraints.'
    )
    parser.add_argument('--corp', type=str, default='constraint',
                        choices=['constraint', 'hybrid'],
                        help='Train VCG gadgets or run HybridQAOA')
    parser.add_argument('--max_cities', type=int, default=3,
                        help='Maximum number of cities')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of QAOA layers (hybrid mode only)')
    parser.add_argument('--results_dir', type=str, default='./results/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    if args.corp == 'constraint':
        run_vcg(args.max_cities,
                result_dir=args.results_dir, data_dir=args.data_dir)
    elif args.corp == 'hybrid':
        run_hybrid(args.max_cities, n_layers=args.n_layers,
                   result_dir=args.results_dir, data_dir=args.data_dir)
    else:
        raise ValueError('--corp must be "constraint" or "hybrid"')
