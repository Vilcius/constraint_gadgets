"""
build_problem_table.py -- Build a per-problem summary DataFrame.

Two entry points:

1. build_problem_table(comp_ar_path, comp_res_path, output_path)
   Classic entry point: reads pre-processed comparison_ar.pkl +
   comparison_resources.pkl (produced by main_analysis.py).

2. build_problem_table_from_raw(raw_pkl_path, data_dir, output_prefix)
   New entry point: reads the raw merged pkl produced by
   run_hybrid_vs_penalty_jax.py, computes all metrics (p_feas, p_opt,
   AR_feas, final-layer selection) inline, and saves both the processed
   comparison_ar pkl and the wide problem_table.

Output columns (both entry points)
-----------------------------------
Problem identity:
    constraints_hash, constraint_type, n_x, n_c, qubo_string,
    has_feasible_solution

Per-method columns (suffix _h = Hybrid, _p = Penalty):
    n_layers_{h,p}          -- final layer reached
    total_time_hr_{h,p}     -- cumulative wall-clock time (hours)
    AR_{h,p}                -- approximation ratio at final layer
    AR_feas_{h,p}           -- feasibility-conditioned AR
    p_feasible_{h,p}        -- P(feasible) at final layer
    p_optimal_{h,p}         -- P(optimal) at final layer
    converged_{h,p}         -- bool: p_feasible >= 0.75
    finished_{h,p}          -- bool: result exists in dataset

Derived:
    max_total_time_hr       -- max(total_time_hr_h, total_time_hr_p)
    min_total_time_hr       -- min of available times

Usage
-----
    cd /home/vilcius/Papers/constraint_gadget/code
    # from pre-processed files (original pipeline):
    python analyze_results/build_problem_table.py \\
        --comp-ar results/overlapping/comparison_ar.pkl \\
        --comp-res results/overlapping/comparison_resources.pkl \\
        --output results/overlapping/problem_table.csv

    # directly from raw jax merged pkl:
    python analyze_results/build_problem_table.py \\
        --raw results/overlapping/hybrid_vs_penalty_jax.pkl \\
        --output results/overlapping/problem_table_jax.csv
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


P_FEAS_THRESHOLD = 0.75  # convergence threshold (mirrors run script)

# Maps full class name -> suffix used in problem_table columns
_METHOD_SUFFIX = {
    'HybridQAOA':          'h',
    'HybridQAOACatalyst':  'h',
    'PenaltyQAOA':         'p',
    'PenaltyQAOACatalyst': 'p',
}


def _unwrap(v):
    """Return v[0] if list-wrapped, else v."""
    return v[0] if isinstance(v, list) else v


def build_problem_table_from_raw(
    raw_pkl_path: str,
    data_dir: str = 'data/',
    output_prefix: str | None = None,
) -> pd.DataFrame:
    """Build a per-problem summary table directly from a raw merged pkl.

    Works with the pkl produced by run_hybrid_vs_penalty_jax.py (one row per
    layer per method per problem).  Computes all metrics inline:

      - p_feasible, p_optimal  — from counts + constraints (no QUBO needed)
      - AR_feas                — feasibility-conditioned AR (needs QUBO;
                                 loaded from qubos.csv via qubo_string)
      - final layer selection  — first layer where p_feas >= 0.75, else last
      - wide pivot             — same column layout as build_problem_table()

    Parameters
    ----------
    raw_pkl_path : str
        Path to the raw merged pkl (e.g. results/overlapping/hybrid_vs_penalty_jax.pkl).
    data_dir : str
        Directory containing qubos.csv (needed for AR_feas computation).
    output_prefix : str or None
        If given, saves:
          {output_prefix}.csv / .pkl          — problem_table (wide)
          {output_prefix}_comparison_ar.pkl   — processed per-method final-layer df

    Returns
    -------
    pd.DataFrame  (one row per problem, wide format)
    """
    from analyze_results.metrics import (
        p_feasible_hybrid, p_optimal_hybrid,
        ar_feasibility_conditioned, aggregate_counts, feasibility_check,
    )

    print(f"Loading raw pkl: {raw_pkl_path}")
    raw = pd.read_pickle(raw_pkl_path)
    print(f"  {len(raw):,} rows")

    # ── Load QUBOs for AR_feas ───────────────────────────────────────────────
    qubo_lookup: dict = {}  # qubo_string -> np.ndarray
    qubo_csv = os.path.join(data_dir, 'qubos.csv')
    if os.path.exists(qubo_csv):
        try:
            from data.make_data import read_qubos_from_file
            all_qubos = read_qubos_from_file('qubos.csv', results_dir=data_dir)
            for n_x_key, idx_dict in all_qubos.items():
                for idx, qd in idx_dict.items():
                    qubo_lookup[qd['qubo_string']] = np.array(qd['Q'])
            print(f"  Loaded {len(qubo_lookup)} QUBOs for AR_feas computation")
        except Exception as e:
            print(f"  [warn] Could not load QUBOs: {e}. AR_feas will be NaN.")

    # ── Compute per-row metrics ──────────────────────────────────────────────
    processed_rows = []
    for _, row in raw.iterrows():
        rd = row.to_dict()

        method_raw = _unwrap(rd.get('method', ''))
        n_x        = int(_unwrap(rd.get('n_x', 0)))
        n_layers   = int(_unwrap(rd.get('n_layers', rd.get('layer', 1))))
        qubo_str   = str(_unwrap(rd.get('qubo_string', '')))

        constraints = _unwrap(rd.get('constraints', []))
        if constraints and isinstance(constraints[0], list):
            constraints = constraints[0]
        constraints_hash = str(sorted(constraints))

        optimal_x = _unwrap(rd.get('optimal_x'))
        if optimal_x and isinstance(optimal_x[0], list):
            optimal_x = optimal_x[0]
        has_feasible = bool(optimal_x) if optimal_x else False

        # AR (from Ising Hamiltonian eigenvalues — already in row)
        opt_cost = _to_float(rd.get('opt_cost'))
        C_max    = _to_float(rd.get('C_max'))
        C_min    = _to_float(rd.get('C_min'))
        ar = (opt_cost - C_max) / (C_min - C_max) if (C_min - C_max) != 0 else float('nan')

        # p_feasible, p_optimal
        p_feas = p_feasible_hybrid(rd)
        p_opt  = p_optimal_hybrid(rd)

        # AR_feas (needs QUBO matrix)
        ar_feas = float('nan')
        if qubo_str in qubo_lookup:
            Q = qubo_lookup[qubo_str]
            counts = _unwrap(rd.get('counts', {}))
            if isinstance(counts, list):
                counts = counts[0]
            min_val = _to_float(rd.get('min_val'))
            if not np.isnan(min_val):
                result = ar_feasibility_conditioned(
                    counts, Q, constraints, n_x, f_star=min_val
                )
                ar_feas = result['AR_feas']

        # Cumulative time (hours)
        cum_time_s = _to_float(rd.get('cumulative_time'))
        total_time_hr = cum_time_s / 3600.0 if not np.isnan(cum_time_s) else float('nan')

        processed_rows.append({
            'method':            method_raw,
            'constraints_hash':  constraints_hash,
            'constraint_type':   str(_unwrap(rd.get('constraint_type', ''))),
            'n_x':               n_x,
            'n_c':               int(_unwrap(rd.get('n_c', len(constraints)))),
            'qubo_string':       qubo_str,
            'has_feasible_solution': has_feasible,
            'n_layers':          n_layers,
            'total_time_hr':     total_time_hr,
            'AR':                ar,
            'AR_feas':           ar_feas,
            'p_feasible':        p_feas,
            'p_optimal':         p_opt,
        })

    proc = pd.DataFrame(processed_rows)

    # ── Select final layer per (method, problem) ─────────────────────────────
    # "Final" = first layer where p_feas >= threshold, or the last layer tried.
    id_cols = ['method', 'constraints_hash', 'qubo_string']
    final_rows = []
    for key, grp in proc.groupby(id_cols, sort=False):
        grp = grp.sort_values('n_layers')
        converged = grp[grp['p_feasible'] >= P_FEAS_THRESHOLD]
        final = converged.iloc[0] if not converged.empty else grp.iloc[-1]
        final_rows.append(final.to_dict())

    final_df = pd.DataFrame(final_rows)
    final_df['converged'] = final_df['p_feasible'] >= P_FEAS_THRESHOLD
    final_df['finished']  = True

    print(f"  Final-layer df: {len(final_df):,} rows "
          f"({final_df['method'].value_counts().to_dict()})")

    # Save comparison_ar-style pkl if requested
    if output_prefix:
        ar_path = output_prefix + '_comparison_ar.pkl'
        os.makedirs(os.path.dirname(ar_path) if os.path.dirname(ar_path) else '.', exist_ok=True)
        final_df.to_pickle(ar_path)
        print(f"  Saved comparison_ar → {ar_path}")

    # ── Pivot to wide format ─────────────────────────────────────────────────
    problem_cols = ['constraints_hash', 'constraint_type', 'n_x', 'n_c',
                    'qubo_string', 'has_feasible_solution']
    metric_cols  = ['n_layers', 'total_time_hr', 'AR', 'AR_feas',
                    'p_feasible', 'p_optimal', 'converged', 'finished']

    frames = {}
    for method_name, suffix in _METHOD_SUFFIX.items():
        sub = final_df[final_df['method'] == method_name].copy()
        if sub.empty:
            continue
        rename = {c: f'{c}_{suffix}' for c in metric_cols}
        frames[suffix] = (sub[problem_cols + metric_cols]
                          .rename(columns=rename)
                          .drop_duplicates(subset=problem_cols))

    if not frames:
        print("  No method data found — returning empty DataFrame.")
        return pd.DataFrame()

    result = None
    for suffix, frame in frames.items():
        result = frame if result is None else result.merge(frame, on=problem_cols, how='outer')

    for suffix in set(_METHOD_SUFFIX.values()):
        col = f'finished_{suffix}'
        if col in result.columns:
            result[col] = result[col].fillna(False)

    time_cols = [f'total_time_hr_{s}' for s in set(_METHOD_SUFFIX.values())
                 if f'total_time_hr_{s}' in result.columns]
    if time_cols:
        result['max_total_time_hr'] = result[time_cols].max(axis=1)
        result['min_total_time_hr'] = result[time_cols].min(axis=1)

    result = result.sort_values('max_total_time_hr', na_position='last').reset_index(drop=True)
    print(f"\nProblem table: {len(result):,} rows x {len(result.columns)} columns")

    if output_prefix:
        csv_path = output_prefix + '.csv'
        pkl_path = output_prefix + '.pkl'
        result.to_csv(csv_path, index=False)
        result.to_pickle(pkl_path)
        print(f"  Saved CSV → {csv_path}")
        print(f"  Saved pkl → {pkl_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw',      default=None,
                        help='Raw merged pkl from run_hybrid_vs_penalty_jax.py '
                             '(skips --comp-ar / --comp-res if given)')
    parser.add_argument('--data-dir', default='data/')
    parser.add_argument('--comp-ar',  default='results/overlapping/comparison_ar.pkl')
    parser.add_argument('--comp-res', default='results/overlapping/comparison_resources.pkl')
    parser.add_argument('--output',   default='results/overlapping/problem_table.csv')
    args = parser.parse_args()

    if args.raw:
        prefix = args.output.replace('.csv', '') if args.output.endswith('.csv') else args.output
        build_problem_table_from_raw(args.raw, data_dir=args.data_dir, output_prefix=prefix)
        return

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
