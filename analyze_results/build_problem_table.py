"""
build_problem_table.py -- Build a per-problem summary DataFrame.

Each row represents one unique problem (constraint set + QUBO), with paired
columns for HybridQAOA and PenaltyQAOA so everything is in one place for
plotting and Catalyst benchmark selection.

Output columns
--------------
Problem identity:
    constraints_hash, constraint_type, n_x, n_c, qubo_string,
    has_feasible_solution

Per-method columns (suffix _h = HybridQAOA, _p = PenaltyQAOA):
    n_layers_{h,p}          -- final layer reached
    total_time_hr_{h,p}     -- total wall-clock time (hours), from cumulative_time
    AR_{h,p}                -- approximation ratio at final layer
    AR_feas_{h,p}           -- feasibility-conditioned AR
    p_feasible_{h,p}        -- P(feasible) at final layer
    p_optimal_{h,p}         -- P(optimal) at final layer
    converged_{h,p}         -- bool: p_feasible >= 0.75
    finished_{h,p}          -- bool: result exists in dataset

Derived:
    max_total_time_hr       -- max(total_time_hr_h, total_time_hr_p); NaN if both missing
    min_total_time_hr       -- min of available times

Usage
-----
    cd /home/vilcius/Papers/constraint_gadget/code
    python analyze_results/build_problem_table.py [--output results/problem_table.csv]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _to_float(v):
    if v is None:
        return float('nan')
    if isinstance(v, list):
        v = v[-1]
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def build_problem_table(
    comp_ar_path: str,
    comp_res_path: str,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Load split pickles and build the per-problem wide DataFrame.

    Parameters
    ----------
    comp_ar_path : str
        Path to comparison_ar.pkl (one row per experiment, final layer).
    comp_res_path : str
        Path to comparison_resources.pkl (all layers, for time aggregation).
    output_path : str or None
        If given, save the resulting DataFrame as a CSV (and pickle).

    Returns
    -------
    pd.DataFrame  (one row per problem)
    """
    print(f"Loading {comp_ar_path} ...")
    ar = pd.read_pickle(comp_ar_path)
    print(f"  {len(ar):,} rows, methods: {ar['method'].value_counts().to_dict()}")

    print(f"Loading {comp_res_path} ...")
    res = pd.read_pickle(comp_res_path)
    print(f"  {len(res):,} rows")

    # ------------------------------------------------------------------
    # Compute total solve time from resources
    # (max cumulative_time across layers per experiment)
    # ------------------------------------------------------------------
    id_cols = [c for c in ['method', 'qubo_string', 'constraints_hash',
                            'n_x', 'angle_strategy']
               if c in res.columns]

    res = res.copy()
    # Coerce all time columns to numeric (handles object-dtype NaN, nested lists, None)
    for c in ['optimize_time', 'counts_time', 'hamiltonian_time', 'cumulative_time']:
        if c in res.columns:
            res[c] = pd.to_numeric(
                res[c].apply(lambda v: v[-1] if isinstance(v, list) else v),
                errors='coerce',
            ).fillna(0.0)

    if 'cumulative_time' in res.columns and res['cumulative_time'].gt(0).any():
        task_times = (
            res.groupby(id_cols)['cumulative_time']
            .max()
            .reset_index()
            .rename(columns={'cumulative_time': 'total_time_s'})
        )
    else:
        # Fall back to summing per-layer component times
        for c in ['optimize_time', 'counts_time', 'hamiltonian_time']:
            if c not in res.columns:
                res[c] = 0.0
        res['_row_time'] = res['optimize_time'] + res['counts_time'] + res['hamiltonian_time']
        task_times = (
            res.groupby(id_cols)['_row_time']
            .sum()
            .reset_index()
            .rename(columns={'_row_time': 'total_time_s'})
        )

    task_times['total_time_hr'] = task_times['total_time_s'] / 3600.0

    # Merge total_time into AR dataframe
    merge_cols = [c for c in ['method', 'qubo_string', 'constraints_hash', 'n_x', 'angle_strategy']
                  if c in ar.columns and c in task_times.columns]
    ar = ar.merge(task_times[merge_cols + ['total_time_hr']], on=merge_cols, how='left')

    # ------------------------------------------------------------------
    # Pivot to wide format: one row per problem, columns per method
    # ------------------------------------------------------------------
    method_map = {'HybridQAOA': 'h', 'PenaltyQAOA': 'p'}

    # Problem identity columns (same across both methods)
    problem_cols = [c for c in [
        'constraints_hash', 'constraint_type', 'n_x', 'n_c',
        'qubo_string', 'has_feasible_solution',
    ] if c in ar.columns]

    # Metric columns to pivot
    metric_cols = [c for c in [
        'n_layers', 'total_time_hr',
        'AR', 'AR_feas', 'p_feasible', 'p_optimal',
        'opt_cost', 'C_max', 'C_min', 'min_val',
    ] if c in ar.columns]

    # Build per-method subsets
    frames = {}
    for method, suffix in method_map.items():
        sub = ar[ar['method'] == method].copy()
        if sub.empty:
            continue
        sub['converged'] = sub['p_feasible'] >= 0.75 if 'p_feasible' in sub.columns else False
        sub['finished'] = True

        rename = {c: f'{c}_{suffix}' for c in metric_cols + ['converged', 'finished']}
        frames[suffix] = sub[problem_cols + list(rename.keys())].rename(columns=rename)

    # Outer-join on problem identity
    if not frames:
        print("  No method data found.")
        return pd.DataFrame()

    result = None
    for suffix, frame in frames.items():
        if result is None:
            result = frame
        else:
            result = result.merge(frame, on=problem_cols, how='outer')

    # Fill missing finished flags with False
    for suffix in method_map.values():
        col = f'finished_{suffix}'
        if col in result.columns:
            result[col] = result[col].fillna(False)

    # Derived columns
    time_cols = [f'total_time_hr_{s}' for s in method_map.values()
                 if f'total_time_hr_{s}' in result.columns]
    if time_cols:
        result['max_total_time_hr'] = result[time_cols].max(axis=1)
        result['min_total_time_hr'] = result[time_cols].min(axis=1)

    result = result.sort_values('max_total_time_hr', na_position='last').reset_index(drop=True)

    print(f"\nProblem table: {len(result):,} rows x {len(result.columns)} columns")
    print(f"  Columns: {result.columns.tolist()}")
    print(f"\n  Time distribution:")
    if 'max_total_time_hr' in result.columns:
        mt = result['max_total_time_hr'].dropna()
        for lo, hi, label in [(0,1,'<1hr'), (1,6,'1-6hr'), (6,12,'6-12hr'),
                               (12,24,'12-24hr'), (24,72,'1-3 days'), (72,1e9,'>3 days')]:
            n = int(((mt >= lo) & (mt < hi)).sum())
            print(f"    {label:10s}: {n}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if output_path:
        csv_path = output_path if output_path.endswith('.csv') else output_path + '.csv'
        pkl_path = csv_path.replace('.csv', '.pkl')
        result.to_csv(csv_path, index=False)
        result.to_pickle(pkl_path)
        print(f"\n  Saved CSV → {csv_path}")
        print(f"  Saved pkl → {pkl_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--comp-ar',  default='results/overlapping/comparison_ar.pkl')
    parser.add_argument('--comp-res', default='results/overlapping/comparison_resources.pkl')
    parser.add_argument('--output',   default='results/overlapping/problem_table.csv')
    args = parser.parse_args()

    tbl = build_problem_table(args.comp_ar, args.comp_res, args.output)

    # Print a sample from each time bracket for quick inspection
    if not tbl.empty and 'max_total_time_hr' in tbl.columns:
        for lo, hi, label in [(1,6,'1-6hr'), (12,24,'12-24hr'), (24,72,'1-3 days')]:
            sub = tbl[(tbl['max_total_time_hr'] >= lo) & (tbl['max_total_time_hr'] < hi)]
            if sub.empty:
                print(f"\n=== {label}: no experiments ===")
                continue
            show_cols = [c for c in [
                'n_x', 'constraint_type', 'max_total_time_hr',
                'n_layers_h', 'total_time_hr_h', 'finished_h', 'converged_h',
                'n_layers_p', 'total_time_hr_p', 'finished_p', 'converged_p',
                'AR_feas_h', 'AR_feas_p', 'p_feasible_h', 'p_feasible_p',
                'constraints_hash',
            ] if c in sub.columns]
            print(f"\n=== {label} ({len(sub)} problems) — sample ===")
            print(sub[show_cols].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
