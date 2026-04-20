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

import itertools

from analyze_results.metrics import (
    p_feasible_hybrid, p_optimal_hybrid,
    ar_feasibility_conditioned,
)


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_ALWAYS_DROP = ['Hamiltonian', 'opt_angles', 'task']

_SHARED_META = [
    'method', 'qubo_string', 'constraint_type', 'n_x', 'n_c',
    'angle_strategy', 'n_layers', 'layer',
]

# Hybrid vs Penalty splits
COMPARISON_AR_COLS = _SHARED_META + [
    'constraints_hash', 'mixer', 'penalty',
    'AR', 'AR_feas', 'p_feasible', 'p_optimal',
    'min_val', 'C_max', 'C_min', 'opt_cost',
    'has_feasible_solution',
]
COMPARISON_RESOURCES_COLS = _SHARED_META + [
    'constraints_hash', 'mixer', 'num_gamma', 'num_beta',
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
    """Pickle *df* to *path* and print a one-line summary (row count, file size).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    path : str
        Destination file path (parent directory must already exist).
    label : str
        Short description printed in the summary line, e.g. ``'VCG AR'``.
    """
    df.to_pickle(path)
    mb = os.path.getsize(path) / 1e6
    print(f"  {label:30s} → {os.path.basename(path)}  ({len(df):,} rows, {mb:.1f} MB)")


def _load_glob(pattern: str) -> pd.DataFrame:
    """Load all pickle files matching *pattern* and concatenate into one DataFrame.

    Files that cannot be read (corrupt, wrong type) are skipped with a warning.

    Parameters
    ----------
    pattern : str
        Shell glob pattern, e.g. ``'gadgets/pending/task_*.pkl'``.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame, or an empty DataFrame if no files matched or
        all files failed to load.
    """
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

def process_vcg(noflag_db_path: str, output_dir: str) -> None:
    """Load noflag_db.pkl (the curated gadget database) and save split files.

    noflag_db is a dict keyed by constraint string whose values contain
    n_layers, ar, entropy, n_x, and train_time.  It is the authoritative
    source for VCG statistics: it includes both trained gadgets (n_layers >= 1)
    and exact-prep gadgets (n_layers == 0, Dicke-structure feasible sets).

    Writes two pickles to *output_dir*:

    * ``vcg_ar.pkl``        — AR, feasibility, entropy columns (small)
    * ``vcg_resources.pkl`` — n_layers, train_time columns (small)

    Parameters
    ----------
    noflag_db_path : str
        Path to the noflag gadget database pickle (e.g. ``gadgets/noflag_db.pkl``).
    output_dir : str
        Destination directory for the split pickles.
    """
    print(f"\n{'='*60}")
    print("  VCG results (noflag_db)")
    print(f"{'='*60}")

    if not os.path.exists(noflag_db_path):
        print(f"  File not found: {noflag_db_path}")
        return

    import pickle as _pickle
    with open(noflag_db_path, 'rb') as f:
        db = _pickle.load(f)

    rows = []
    for constraint_str, v in db.items():
        fam = ('quadratic_knapsack'
               if re.search(r'x_\d+\*x_\d+', constraint_str)
               else 'knapsack')
        rows.append({
            'constraint_type': fam,
            'constraints':     v.get('constraints', [constraint_str]),
            'n_x':             v.get('n_x'),
            'n_layers':        v.get('n_layers'),
            'AR':              v.get('ar'),
            'entropy':         v.get('entropy'),
            'train_time':      v.get('train_time'),
        })

    df = pd.DataFrame(rows)
    print(f"  {len(df):,} gadgets loaded")
    print(f"  n_layers dist: {df['n_layers'].value_counts().sort_index().to_dict()}")

    ar_cols   = ['constraint_type', 'constraints', 'n_x', 'n_layers', 'AR', 'entropy']
    res_cols  = ['constraint_type', 'n_x', 'n_layers', 'train_time']
    _save(_select(df, ar_cols),  os.path.join(output_dir, 'vcg_ar.pkl'),        'VCG AR')
    _save(_select(df, res_cols), os.path.join(output_dir, 'vcg_resources.pkl'), 'VCG Resources')


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


def _check_has_feasible(row) -> bool:
    """Brute-force check whether any bitstring of length n_x satisfies all constraints."""
    constraints = row['constraints']
    if isinstance(constraints, list) and constraints and isinstance(constraints[0], list):
        constraints = constraints[0]
    n_x = int(row['n_x'][0] if isinstance(row['n_x'], list) else row['n_x'])
    for bits in itertools.product([0, 1], repeat=n_x):
        var_dict = {f'x_{i}': b for i, b in enumerate(bits)}
        if all(eval(c, {"__builtins__": {}}, var_dict) for c in constraints):
            return True
    return False


def process_hybrid(hybrid_path: str, output_dir: str, save_counts: bool) -> None:
    """Load the merged hybrid vs penalty results pickle, compute metrics, and save splits.

    Steps performed:

    1. Load *hybrid_path* and unpack any length-1 list columns.
    2. Dedup step 1: drop exact duplicate saves at the same layer on
       ``(method, qubo_string, constraints_hash, n_x, layer, angle_strategy)``.
    3. Compute ``p_feasible``, ``p_optimal``, and ``AR_feas`` (the last requires
       the QUBO matrix loaded from ``data/qubos.csv``).
    4. Dedup step 2 (AR only): keep the **final layer** per experiment
       (group by experiment identity, take max-layer row).  Each experiment
       iteratively adds QAOA layers, saving a row after each; only the last row
       reflects the completed experiment state.  Resources keeps all rows so
       that per-layer times can be summed for total-time calculations.
    5. Write up to three split pickles to *output_dir*:

       * ``comparison_ar.pkl``        — one row per experiment (final layer)
       * ``comparison_resources.pkl`` — all layer rows (for time aggregation)
       * ``comparison_counts.pkl``    — raw measurement counts (large; optional)

    Parameters
    ----------
    hybrid_path : str
        Path to the merged ``hybrid_vs_penalty.pkl`` produced by the run scripts.
    output_dir : str
        Destination directory for the split pickles.
    save_counts : bool
        If ``False``, skip writing ``comparison_counts.pkl`` (saves disk space).
    """
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

    # Normalise JAX class names to canonical names used by plot scripts
    _METHOD_REMAP = {'HybridQAOACatalyst': 'HybridQAOA', 'PenaltyQAOACatalyst': 'PenaltyQAOA'}
    if 'method' in df.columns:
        df['method'] = df['method'].map(lambda m: _METHOD_REMAP.get(m, m))

    print(f"  {len(df):,} rows loaded ({df['method'].value_counts().to_dict() if 'method' in df.columns else ''})")

    # Dedup step 1: drop exact duplicate saves at the same layer.
    # constraints_hash distinguishes instances with same qubo_string but different coefficients.
    if 'constraints' in df.columns:
        df['constraints_hash'] = df['constraints'].apply(lambda c: str(c))
    dedup_keys = [c for c in ['method', 'constraint_type', 'qubo_string', 'constraints_hash', 'n_x', 'layer', 'angle_strategy']
                  if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep='first').reset_index(drop=True)
    if len(df) < before:
        print(f"  Removed {before - len(df):,} duplicate saves at the same layer")

    # Compute derived metrics (requires counts)
    df['p_feasible'] = df.apply(p_feasible_hybrid, axis=1)
    df['p_optimal']  = df.apply(p_optimal_hybrid, axis=1)

    # AR_feas: feasibility-conditioned AR.  Requires the QUBO matrix, which is
    # not stored directly — load it from data/qubos.csv keyed by qubo_string.
    df['AR_feas'] = _compute_ar_feas_column(df)

    # has_feasible_solution: brute-force check; True iff at least one bitstring
    # satisfies all constraints.  n_x <= 10 so at most 1024 evaluations per row.
    # Computed once per unique constraint_hash then broadcast to all rows.
    if 'constraints' in df.columns and 'constraints_hash' in df.columns:
        unique = df.drop_duplicates(subset=['constraints_hash'])[
            ['constraints_hash', 'constraints', 'n_x']
        ]
        feas_map = {
            row['constraints_hash']: _check_has_feasible(row)
            for _, row in unique.iterrows()
        }
        df['has_feasible_solution'] = df['constraints_hash'].map(feas_map)
        n_infeasible = int((~df['has_feasible_solution']).sum())
        unique_counts = (unique['constraints_hash']
                         .map(feas_map).value_counts().to_dict())
        print(f"  Instances with no feasible solution: "
              f"{n_infeasible}/{len(df)} rows "
              f"(unique constraint sets — {unique_counts})")
    else:
        df['has_feasible_solution'] = True

    # Dedup step 2 (AR only): keep the final layer per experiment.
    # Each experiment iteratively adds QAOA layers until convergence or p_max,
    # saving a row after each layer.  For AR/feasibility analysis only the final
    # row (highest layer) matters; resources keeps all rows for time aggregation.
    exp_keys = [c for c in ['method', 'qubo_string', 'constraints_hash', 'n_x', 'angle_strategy']
                if c in df.columns]
    layer_col = 'layer' if 'layer' in df.columns else 'n_layers'
    df_ar = (df.sort_values(layer_col)
               .groupby(exp_keys, sort=False)
               .last()
               .reset_index())
    n_experiments = len(df_ar)
    print(f"  Experiments (final-layer rows): {n_experiments} "
          f"({df_ar.groupby('method').size().to_dict() if 'method' in df_ar.columns else ''})")

    _save(_select(df_ar, COMPARISON_AR_COLS),
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
    """CLI entry point.  Parse arguments and run :func:`process_vcg` and
    :func:`process_hybrid`."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--noflag-db',  default='gadgets/noflag_db.pkl',
                        help='No-flag VCG gadget database pickle (default: gadgets/noflag_db.pkl)')
    parser.add_argument('--hybrid',     default='results/overlapping/hybrid_vs_penalty.pkl',
                        help='Merged hybrid vs penalty results pickle')
    parser.add_argument('--output-dir', default='results/',
                        help='Output directory for split pickles (default: results/)')
    parser.add_argument('--no-counts',  action='store_true',
                        help='Skip the large comparison_counts.pkl')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_vcg(args.noflag_db, args.output_dir)
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
