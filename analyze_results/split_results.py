"""
split_results.py -- Load raw result pickles and split into purpose-specific
DataFrames, computing derived metrics (p_feasible, p_optimal, depth) in the
process so downstream analysis scripts do not need raw counts or Hamiltonians.

Produces (in --output-dir, default: results/):
    vcg_ar.pkl            -- VCG: AR, p_feasible, metadata            (small)
    vcg_resources.pkl     -- VCG: shots, depth, timing, metadata      (small)
    comparison_ar.pkl     -- Hybrid vs Penalty: AR, p_feasible,
                             p_optimal, metadata                       (small)
    comparison_resources.pkl -- Hybrid vs Penalty: shots, timing,
                             parameter counts, metadata                (small)
    comparison_counts.pkl -- Hybrid vs Penalty: raw counts + constraints
                             (large; skip with --no-counts)

Usage
-----
    python analyze_results/split_results.py
    python analyze_results/split_results.py \\
        --vcg-dir    gadgets/pending/ \\
        --hybrid     results/hybrid_vs_penalty.pkl \\
        --output-dir results/
    python analyze_results/split_results.py --no-counts
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
import pandas as pd

from analyze_results.metrics import (
    p_feasible_vcg, p_feasible_hybrid, p_optimal_hybrid,
    ar_feasibility_conditioned,
)


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_ALWAYS_DROP = ['Hamiltonian', 'opt_angles', 'qubo_string', 'task']

_SHARED_META = [
    'method', 'constraint_type', 'n_x', 'n_c',
    'angle_strategy', 'n_layers', 'layer',
]

# VCG splits
VCG_AR_COLS = [
    'constraint_type', 'constraints', 'n_x', 'n_c',
    'angle_strategy', 'n_layers', 'num_gamma', 'num_beta',
    'AR', 'p_feasible', 'opt_cost', 'C_max', 'C_min',
]
VCG_RESOURCES_COLS = [
    'constraint_type', 'n_x', 'n_c',
    'angle_strategy', 'n_layers', 'num_gamma', 'num_beta',
    'est_shots', 'est_error', 'group_est_shots', 'group_est_error',
    'depth', 'num_gates',
    'hamiltonian_time', 'optimize_time', 'counts_time',
]

# Hybrid vs Penalty splits
COMPARISON_AR_COLS = _SHARED_META + [
    'mixer', 'penalty',
    'AR', 'AR_feas', 'p_feasible', 'p_optimal',
    'min_val', 'C_max', 'C_min', 'opt_cost',
]
COMPARISON_RESOURCES_COLS = _SHARED_META + [
    'mixer', 'num_gamma', 'num_beta',
    'est_shots', 'est_error', 'group_est_shots', 'group_est_error',
    'hamiltonian_time', 'optimize_time', 'counts_time',
]
COMPARISON_COUNTS_COLS = _SHARED_META + [
    'constraints', 'counts',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return df restricted to columns that exist."""
    return df[[c for c in cols if c in df.columns]].copy()


def _save(df: pd.DataFrame, path: str, label: str) -> None:
    df.to_pickle(path)
    mb = os.path.getsize(path) / 1e6
    print(f"  {label:30s} → {os.path.basename(path)}  ({len(df):,} rows, {mb:.1f} MB)")


def _load_glob(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    print(f"  Loading {len(files)} file(s) from {pattern} ...")
    frames = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            if isinstance(df, pd.DataFrame):
                frames.append(df)
            else:
                print(f"    WARNING: {os.path.basename(f)} is not a DataFrame, skipping")
        except Exception as e:
            print(f"    WARNING: could not read {os.path.basename(f)}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _unpack_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Unwrap length-1 list values produced by collect_* functions."""
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, list) and len(v) == 1).all():
            df[col] = df[col].apply(lambda v: v[0])
    return df


def _extract_resources(df: pd.DataFrame) -> pd.DataFrame:
    """Unpack CircuitResources objects into depth / num_gates columns."""
    if 'resources' in df.columns:
        df['depth'] = df['resources'].apply(
            lambda r: r.depth if r is not None else float('nan'))
        df['num_gates'] = df['resources'].apply(
            lambda r: r.num_gates if r is not None else float('nan'))
    return df


# ---------------------------------------------------------------------------
# VCG processing
# ---------------------------------------------------------------------------

def _infer_vcg_constraint_type(row) -> str:
    """Infer constraint family from constraint strings when constraint_type is blank."""
    ct = row.get('constraint_type', '')
    if isinstance(ct, list):
        ct = ct[0] if ct else ''
    if ct and ct != '':
        return ct
    constraints = row.get('constraints', [])
    if isinstance(constraints, list) and constraints:
        c = constraints[0] if isinstance(constraints[0], list) else constraints[0]
        if isinstance(c, list):
            c = c[0] if c else ''
        if re.search(r'x_\d+\*x_\d+', str(c)):
            return 'quadratic_knapsack'
        return 'knapsack'
    return 'unknown'


def process_vcg(vcg_dir: str, output_dir: str) -> None:
    print(f"\n{'='*60}")
    print("  VCG results")
    print(f"{'='*60}")

    df = _load_glob(os.path.join(vcg_dir, 'task_*.pkl'))
    if df.empty:
        print("  No VCG task pickles found.")
        return

    df = _unpack_list_cols(df)
    print(f"  {len(df):,} rows loaded")

    # Infer constraint_type when missing (legacy data from before family was stored)
    df['constraint_type'] = df.apply(_infer_vcg_constraint_type, axis=1)

    # Compute derived metrics
    df['p_feasible'] = df.apply(p_feasible_vcg, axis=1)
    df = _extract_resources(df)

    _save(_select(df, VCG_AR_COLS),
          os.path.join(output_dir, 'vcg_ar.pkl'), 'VCG AR')
    _save(_select(df, VCG_RESOURCES_COLS),
          os.path.join(output_dir, 'vcg_resources.pkl'), 'VCG Resources')


# ---------------------------------------------------------------------------
# Hybrid vs Penalty processing
# ---------------------------------------------------------------------------

def _load_qubo_lookup(data_dir: str = 'data/') -> dict:
    """Return {qubo_string: Q_matrix} mapping for all QUBOs in qubos.csv."""
    from data.make_data import read_qubos_from_file
    try:
        qubos = read_qubos_from_file('qubos.csv', results_dir=data_dir)
        lookup = {}
        for n_dict in qubos.values():
            for entry in n_dict.values():
                qs = entry.get('qubo_string', '').strip()
                if qs:
                    lookup[qs] = entry['Q']
        return lookup
    except Exception as e:
        print(f"  WARNING: could not load QUBO lookup: {e}")
        return {}


def _compute_ar_feas_column(df: pd.DataFrame) -> pd.Series:
    """Compute AR_feas for every row in the hybrid DataFrame.

    Requires columns: counts, qubo_string, constraints, n_x, min_val.
    Returns a Series of AR_feas values (NaN when no feasible shots or QUBO missing).
    """
    qubo_lookup = _load_qubo_lookup()
    if not qubo_lookup:
        print("  WARNING: QUBO lookup empty — AR_feas will be all NaN")
        return pd.Series([float('nan')] * len(df), index=df.index)

    results = []
    for _, row in df.iterrows():
        try:
            counts = row['counts']
            if isinstance(counts, list):
                counts = counts[0]
            qs = row['qubo_string']
            if isinstance(qs, list):
                qs = qs[0]
            qs = qs.strip()
            n_x = int(row['n_x'][0] if isinstance(row['n_x'], list) else row['n_x'])
            min_val = float(row['min_val'][0] if isinstance(row['min_val'], list) else row['min_val'])
            constraints = row['constraints']
            if isinstance(constraints, list) and constraints and isinstance(constraints[0], list):
                constraints = constraints[0]

            Q = qubo_lookup.get(qs)
            if Q is None:
                results.append(float('nan'))
                continue

            res = ar_feasibility_conditioned(counts, Q, constraints, n_x, f_star=min_val)
            results.append(res['AR_feas'])
        except Exception:
            results.append(float('nan'))

    n_computed = sum(1 for v in results if not (v != v))  # not NaN
    print(f"  AR_feas computed for {n_computed}/{len(results)} rows")
    return pd.Series(results, index=df.index)


def process_hybrid(hybrid_path: str, output_dir: str, save_counts: bool) -> None:
    print(f"\n{'='*60}")
    print("  Hybrid vs Penalty results")
    print(f"{'='*60}")

    if not os.path.exists(hybrid_path):
        print(f"  File not found: {hybrid_path}")
        return

    df = pd.read_pickle(hybrid_path)
    if df.empty:
        print("  Results file is empty.")
        return

    df = _unpack_list_cols(df)
    print(f"  {len(df):,} rows loaded ({df['method'].value_counts().to_dict() if 'method' in df.columns else ''})")

    # Dedup: drop exact duplicates on identifying columns
    dedup_keys = [c for c in ['method', 'constraint_type', 'n_x', 'layer', 'angle_strategy']
                  if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep='first').reset_index(drop=True)
    if len(df) < before:
        print(f"  Removed {before - len(df):,} duplicate rows")

    # Compute derived metrics (requires counts)
    df['p_feasible'] = df.apply(p_feasible_hybrid, axis=1)
    df['p_optimal']  = df.apply(p_optimal_hybrid, axis=1)

    # AR_feas: feasibility-conditioned AR.  Requires the QUBO matrix, which is
    # not stored directly — load it from data/qubos.csv keyed by qubo_string.
    df['AR_feas'] = _compute_ar_feas_column(df)

    _save(_select(df, COMPARISON_AR_COLS),
          os.path.join(output_dir, 'comparison_ar.pkl'), 'Comparison AR')
    _save(_select(df, COMPARISON_RESOURCES_COLS),
          os.path.join(output_dir, 'comparison_resources.pkl'), 'Comparison Resources')

    if save_counts:
        _save(_select(df, COMPARISON_COUNTS_COLS),
              os.path.join(output_dir, 'comparison_counts.pkl'), 'Comparison Counts')
    else:
        print(f"  {'Comparison Counts':30s} → skipped (--no-counts)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vcg-dir',    default='gadgets/pending/',
                        help='Directory of per-task VCG pickles (default: gadgets/pending/)')
    parser.add_argument('--hybrid',     default='results/hybrid_vs_penalty.pkl',
                        help='Merged hybrid vs penalty results pickle')
    parser.add_argument('--output-dir', default='results/',
                        help='Output directory for split pickles (default: results/)')
    parser.add_argument('--no-counts',  action='store_true',
                        help='Skip the large comparison_counts.pkl')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_vcg(args.vcg_dir, args.output_dir)
    process_hybrid(args.hybrid, args.output_dir, save_counts=not args.no_counts)

    print(f"\n{'='*60}")
    print(f"  Done. Split files written to: {args.output_dir}")
    print(f"{'='*60}")
    print("  Run analysis with:")
    print(f"    python analyze_results/main_analysis.py \\")
    print(f"        --vcg-ar    {args.output_dir}/vcg_ar.pkl \\")
    print(f"        --vcg-res   {args.output_dir}/vcg_resources.pkl \\")
    print(f"        --comp-ar   {args.output_dir}/comparison_ar.pkl \\")
    print(f"        --comp-res  {args.output_dir}/comparison_resources.pkl \\")
    print(f"        --output-dir ./analysis_output/")


if __name__ == '__main__':
    main()
