"""
catalyst_benchmark.py -- Catalyst vs original timing comparison.

Runs 3 pre-selected problems (one per time bracket) using the Catalyst-compiled
QAOA solvers (HybridQAOACatalyst, PenaltyQAOACatalyst) and compares wall-clock
time against the original HPC results stored in results/overlapping/problem_table.pkl.

Original experiment parameters are reproduced exactly (same n_layers sweep, same
steps/restarts/lr).  JIT warm-up time (first call, which triggers Catalyst
compilation) is recorded separately from subsequent call time so both numbers
are visible.

Usage
-----
    cd /home/vilcius/Papers/constraint_gadget/code
    # Run just experiment 1 (fastest, ~1 hr original):
    python run/catalyst_benchmark.py --exp 1

    # Run all 3 sequentially:
    python run/catalyst_benchmark.py --exp all

    # Dry-run to show configs without running:
    python run/catalyst_benchmark.py --dry-run

Output
------
    results/overlapping/catalyst_benchmark.csv   (timing comparison table)
    results/overlapping/catalyst_benchmark.pkl   (full result rows)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data.make_data import read_qubos_from_file
from core import constraint_handler as ch
from core import hybrid_qaoa_catalyst as hqc
from core import penalty_qaoa_catalyst as pqc
from analyze_results.metrics import p_feasible_hybrid

# ---------------------------------------------------------------------------
# Experiment definitions (constraints and qubo_strings from problem_table.pkl)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        'id': 1,
        'label': '~1hr original',
        'n_x': 5,
        'constraint_type': 'cardinality+cardinality',
        'constraints': ['x_0 + x_2 + x_3 <= 3', 'x_0 + x_2 + x_3 == 2'],
        # HPC original: H=1.09hr (l=1, converged), P=1.08hr (l=1, converged)
        'original_h_hr': 1.093,
        'original_p_hr': 1.081,
        'original_h_layers': 1,
        'original_p_layers': 1,
    },
    {
        'id': 2,
        'label': '~12hr original',
        'n_x': 8,
        'constraint_type': 'cardinality+cardinality',
        'constraints': ['x_0 + x_1 + x_2 + x_3 <= 1', 'x_4 + x_5 + x_6 + x_7 == 3'],
        # HPC original: H=0.94hr (l=1, converged), P=12.01hr (l=3, converged)
        'original_h_hr': 0.939,
        'original_p_hr': 12.006,
        'original_h_layers': 1,
        'original_p_layers': 3,
    },
    {
        'id': 3,
        'label': '~26hr original',
        'n_x': 9,
        'constraint_type': 'knapsack+cardinality',
        'constraints': [
            '4*x_3 + 2*x_8 + 3*x_4 + 3*x_6 + 5*x_2 <= 11',
            'x_8 + x_7 + x_4 + x_5 + x_1 >= 5',
        ],
        # HPC original: H=26.38hr (l=5, not converged), P=24.86hr (l=5, not converged)
        'original_h_hr': 26.380,
        'original_p_hr': 24.859,
        'original_h_layers': 5,
        'original_p_layers': 5,
    },
]

# Hyperparameters from run_hybrid_vs_penalty.py
HYBRID_STEPS         = 50
HYBRID_RESTARTS      = 10
HYBRID_LR            = 0.01
HYBRID_CQAOA_STEPS   = 30
HYBRID_CQAOA_RESTARTS = 10
PENALTY_STEPS        = 50
PENALTY_RESTARTS     = 10
PENALTY_LR           = 0.01
MAX_LAYERS           = 5
P_FEAS_THRESHOLD     = 0.75
SHOTS                = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_qubo(constraints: list[str], n_x: int, data_dir: str = 'data/') -> tuple:
    """Return (Q_matrix, qubo_string, qubo_idx, penalty_weight) for given constraints.

    Loads all QUBOs from qubos.csv and picks those whose qubo_string matches
    the original recorded in the problem_table.pkl for this constraint set.
    Falls back to the first QUBO of size n_x if no exact match is found.
    """
    tbl_path = 'results/overlapping/problem_table.pkl'
    con_key = str(sorted(constraints))
    qubo_string_target = None

    if os.path.exists(tbl_path):
        tbl = pd.read_pickle(tbl_path)
        # Find the row with matching constraints_hash
        matches = tbl[tbl['constraints_hash'] == str(constraints)]
        if matches.empty:
            # Try sorted
            matches = tbl[tbl['constraints_hash'].apply(
                lambda h: str(sorted(eval(h) if isinstance(h, str) else h)) == str(sorted(constraints))
                if h else False)]
        if not matches.empty:
            qubo_string_target = matches.iloc[0]['qubo_string']

    qubos = read_qubos_from_file('qubos.csv', results_dir=data_dir)
    n_dict = qubos.get(n_x, {})

    # Try to match by qubo_string
    if qubo_string_target:
        for idx, entry in n_dict.items():
            if entry.get('qubo_string', '').strip() == qubo_string_target.strip():
                Q = entry['Q']
                total_min = float(np.min(np.linalg.eigh(Q)[0]))
                penalty_weight = float(5 + 2 * abs(total_min))
                return Q, entry['qubo_string'], idx, penalty_weight

    # Fallback: first QUBO of the right size
    print(f"  WARNING: qubo_string not matched, using qubo_idx=0 for n_x={n_x}")
    entry = n_dict[0]
    Q = entry['Q']
    total_min = float(np.min(np.linalg.eigh(Q)[0]))
    penalty_weight = float(5 + 2 * abs(total_min))
    return Q, entry['qubo_string'], 0, penalty_weight


def _pfeas(counts: dict, constraints: list, n_x: int) -> float:
    """P(feasible) from a counts dict."""
    # Build a fake 'row' compatible with p_feasible_hybrid
    class _Row:
        pass
    row = {'counts': counts, 'constraints': constraints, 'n_x': n_x}
    from analyze_results.metrics import p_feasible_hybrid as _pfh

    class RowDict(dict):
        def __getitem__(self, k):
            return super().__getitem__(k)

    return _pfh(row)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_experiment(exp: dict, data_dir: str = 'data/',
                   gadget_db_path: str = 'gadgets/noflag_db.pkl') -> dict:
    """Run one benchmark experiment and return timing + quality results."""
    print(f"\n{'='*70}")
    print(f"Experiment {exp['id']}: {exp['label']}")
    print(f"  n_x={exp['n_x']}  constraints={exp['constraints']}")
    print(f"  Original HPC: H={exp['original_h_hr']:.2f}hr  P={exp['original_p_hr']:.2f}hr")
    print(f"{'='*70}")

    constraints_str = exp['constraints']
    n_x = exp['n_x']

    Q, qubo_string, qubo_idx, penalty_weight = _load_qubo(constraints_str, n_x, data_dir)
    print(f"  QUBO loaded: n_x={n_x} qubo_idx={qubo_idx} penalty_weight={penalty_weight:.2f}")

    parsed = ch.parse_constraints(constraints_str)
    structural_indices, penalty_indices = ch.partition_constraints(parsed, strategy='auto')

    result_rows = []

    # -----------------------------------------------------------------------
    # HybridQAOACatalyst sweep
    # -----------------------------------------------------------------------
    print(f"\n  -- HybridQAOACatalyst --")
    prev_h_angles = None
    h_cumulative = 0.0
    h_jit_time = None   # JIT warm-up time (first call)

    for p in range(1, MAX_LAYERS + 1):
        t0 = time.time()
        solver = hqc.HybridQAOACatalyst(
            qubo=Q,
            all_constraints=parsed,
            structural_indices=structural_indices,
            penalty_indices=penalty_indices,
            penalty_str=[penalty_weight],
            penalty_pen=penalty_weight,
            angle_strategy='ma-QAOA',
            mixer='Grover',
            n_layers=p,
            steps=HYBRID_STEPS,
            num_restarts=HYBRID_RESTARTS,
            learning_rate=HYBRID_LR,
            cqaoa_steps=HYBRID_CQAOA_STEPS,
            cqaoa_num_restarts=HYBRID_CQAOA_RESTARTS,
            gadget_db_path=gadget_db_path if os.path.exists(gadget_db_path) else None,
        )
        init_time = time.time() - t0

        opt_cost, opt_angles = solver.optimize_angles(prev_layer_angles=prev_h_angles)
        opt_time = solver.optimize_time

        counts = solver.do_counts_circuit(shots=SHOTS)
        count_time = solver.count_time

        layer_time = init_time + opt_time + count_time
        h_cumulative += layer_time

        if h_jit_time is None and p == 1:
            # First call includes JIT compilation
            h_jit_time = layer_time
            print(f"    p={p}: JIT+opt={layer_time/3600:.4f}hr  cumulative={h_cumulative/3600:.4f}hr")
        else:
            print(f"    p={p}: opt={layer_time/3600:.4f}hr  cumulative={h_cumulative/3600:.4f}hr")

        pfeas = _pfeas(counts, constraints_str, n_x)
        print(f"         P(feas)={pfeas:.4f}  opt_cost={opt_cost:.4f}")

        result_rows.append({
            'exp_id': exp['id'],
            'method': 'HybridQAOACatalyst',
            'n_x': n_x,
            'constraint_type': exp['constraint_type'],
            'n_layers': p,
            'layer_time_s': layer_time,
            'cumulative_time_s': h_cumulative,
            'cumulative_time_hr': h_cumulative / 3600,
            'jit_on_this_layer': (p == 1),
            'p_feasible': pfeas,
            'opt_cost': opt_cost,
            'original_h_hr': exp['original_h_hr'],
            'original_p_hr': exp['original_p_hr'],
        })

        prev_h_angles = np.array(opt_angles)
        if pfeas >= P_FEAS_THRESHOLD:
            print(f"    ✓ converged at p={p}")
            break

    h_total_hr = h_cumulative / 3600
    print(f"  HybridQAOACatalyst total: {h_total_hr:.4f}hr  (original: {exp['original_h_hr']:.2f}hr)")
    speedup_h = exp['original_h_hr'] / h_total_hr if h_total_hr > 0 else float('nan')
    print(f"  Speedup vs HPC: {speedup_h:.2f}x  (laptop vs HPC — not apples-to-apples)")

    # -----------------------------------------------------------------------
    # PenaltyQAOACatalyst sweep
    # -----------------------------------------------------------------------
    print(f"\n  -- PenaltyQAOACatalyst --")
    prev_p_angles = None
    p_cumulative = 0.0
    p_jit_time = None

    for p in range(1, MAX_LAYERS + 1):
        t0 = time.time()
        solver = pqc.PenaltyQAOACatalyst(
            qubo=Q,
            constraints=constraints_str,
            penalty=penalty_weight,
            angle_strategy='ma-QAOA',
            n_layers=p,
            steps=PENALTY_STEPS,
            num_restarts=PENALTY_RESTARTS,
            learning_rate=PENALTY_LR,
        )
        init_time = time.time() - t0

        opt_cost, opt_angles = solver.optimize_angles(prev_layer_angles=prev_p_angles)
        opt_time = solver.optimize_time

        counts = solver.do_counts_circuit(shots=SHOTS)
        count_time = solver.count_time

        layer_time = init_time + opt_time + count_time
        p_cumulative += layer_time

        if p_jit_time is None and p == 1:
            p_jit_time = layer_time
            print(f"    p={p}: JIT+opt={layer_time/3600:.4f}hr  cumulative={p_cumulative/3600:.4f}hr")
        else:
            print(f"    p={p}: opt={layer_time/3600:.4f}hr  cumulative={p_cumulative/3600:.4f}hr")

        pfeas = _pfeas(counts, constraints_str, n_x)
        print(f"         P(feas)={pfeas:.4f}  opt_cost={opt_cost:.4f}")

        result_rows.append({
            'exp_id': exp['id'],
            'method': 'PenaltyQAOACatalyst',
            'n_x': n_x,
            'constraint_type': exp['constraint_type'],
            'n_layers': p,
            'layer_time_s': layer_time,
            'cumulative_time_s': p_cumulative,
            'cumulative_time_hr': p_cumulative / 3600,
            'jit_on_this_layer': (p == 1),
            'p_feasible': pfeas,
            'opt_cost': opt_cost,
            'original_h_hr': exp['original_h_hr'],
            'original_p_hr': exp['original_p_hr'],
        })

        prev_p_angles = np.array(opt_angles)
        if pfeas >= P_FEAS_THRESHOLD:
            print(f"    ✓ converged at p={p}")
            break

    p_total_hr = p_cumulative / 3600
    print(f"  PenaltyQAOACatalyst total: {p_total_hr:.4f}hr  (original: {exp['original_p_hr']:.2f}hr)")
    speedup_p = exp['original_p_hr'] / p_total_hr if p_total_hr > 0 else float('nan')
    print(f"  Speedup vs HPC: {speedup_p:.2f}x  (laptop vs HPC — not apples-to-apples)")

    return result_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp', default='1',
                        help='Experiment ID to run: 1, 2, 3, or "all" (default: 1)')
    parser.add_argument('--data-dir', default='data/')
    parser.add_argument('--db', default='gadgets/noflag_db.pkl')
    parser.add_argument('--output', default='results/overlapping/catalyst_benchmark')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without running')
    args = parser.parse_args()

    # Select experiments to run
    if args.exp == 'all':
        exps = EXPERIMENTS
    else:
        try:
            ids = [int(x.strip()) for x in args.exp.split(',')]
            exps = [e for e in EXPERIMENTS if e['id'] in ids]
        except ValueError:
            parser.error(f'--exp must be 1, 2, 3, or "all", got: {args.exp}')

    if not exps:
        parser.error(f'No matching experiments for --exp={args.exp}')

    if args.dry_run:
        for e in exps:
            print(f"\nExperiment {e['id']}: {e['label']}")
            print(f"  n_x={e['n_x']}  constraints={e['constraints']}")
            print(f"  Original HPC: H={e['original_h_hr']:.2f}hr  P={e['original_p_hr']:.2f}hr")
            print(f"  Steps={HYBRID_STEPS}  Restarts={HYBRID_RESTARTS}  LR={HYBRID_LR}")
            print(f"  MAX_LAYERS={MAX_LAYERS}  SHOTS={SHOTS:,}")
        return

    all_rows = []
    for exp in exps:
        rows = run_experiment(exp, data_dir=args.data_dir, gadget_db_path=args.db)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for exp_id, grp in df.groupby('exp_id'):
        exp = next(e for e in EXPERIMENTS if e['id'] == exp_id)
        print(f"\nExp {exp_id}: {exp['label']}")
        for method, mgrp in grp.groupby('method'):
            total = mgrp['cumulative_time_hr'].max()
            orig = exp['original_h_hr'] if 'Hybrid' in method else exp['original_p_hr']
            speedup = orig / total if total > 0 else float('nan')
            final_pfeas = mgrp.loc[mgrp['n_layers'].idxmax(), 'p_feasible']
            print(f"  {method:30s}: {total:.4f}hr  (orig={orig:.2f}hr  speedup={speedup:.2f}x  pfeas={final_pfeas:.3f})")
        print(f"  Note: HPC vs laptop comparison — hardware difference expected")

    # Save
    out = args.output
    df.to_csv(out + '.csv', index=False)
    df.to_pickle(out + '.pkl')
    print(f"\nResults saved to {out}.csv / {out}.pkl")


if __name__ == '__main__':
    main()
