"""
metrics.py -- Core metric functions for VCG and HybridQAOA results.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-row metric functions
# ---------------------------------------------------------------------------

def p_feasible_vcg(row) -> float:
    """Fraction of counts where the flag bit(s) are all '0' (feasible).

    VCG convention: last n_c bits are flag bits; flag=0 means satisfied.
    """
    counts = row['counts']
    n_c = int(row.get('n_c', 1))
    total = sum(counts.values())
    if total == 0:
        return float('nan')
    feasible = sum(v for k, v in counts.items() if k[-n_c:] == '0' * n_c)
    return feasible / total


def p_feasible_hybrid(row) -> float:
    """Fraction of counts (first n_x bits) satisfying all constraints.

    Uses eval-based feasibility check consistent with data.make_data.get_optimal_x.
    """
    counts = row['counts']
    n_x = int(row['n_x'])
    constraints = row['constraints']
    total = sum(counts.values())
    if total == 0:
        return float('nan')
    feasible = 0
    for bitstring, cnt in counts.items():
        x_bits = bitstring[:n_x]
        var_dict = {f'x_{i}': int(b) for i, b in enumerate(x_bits)}
        if all(eval(c, {"__builtins__": {}}, var_dict) for c in constraints):
            feasible += cnt
    return feasible / total


def p_optimal_hybrid(row) -> float:
    """Fraction of counts achieving min_val (constrained optimum).

    Returns NaN if min_val is None or missing.
    """
    min_val = row.get('min_val')
    if min_val is None or (isinstance(min_val, float) and np.isnan(min_val)):
        return float('nan')
    counts = row['counts']
    n_x = int(row['n_x'])
    constraints = row['constraints']
    total = sum(counts.values())
    if total == 0:
        return float('nan')
    optimal = 0
    for bitstring, cnt in counts.items():
        x_bits = bitstring[:n_x]
        x = [int(b) for b in x_bits]
        var_dict = {f'x_{i}': x[i] for i in range(n_x)}
        # feasibility check
        if not all(eval(c, {"__builtins__": {}}, var_dict) for c in constraints):
            continue
        # use qubo from row if available, else skip objective check
        qubo = row.get('qubo')
        if qubo is not None:
            obj = float(np.dot(x, np.dot(qubo, x)))
            if abs(obj - float(min_val)) < 1e-6:
                optimal += cnt
        else:
            # fallback: no qubo stored; skip
            pass
    return optimal / total if total > 0 else float('nan')


# ---------------------------------------------------------------------------
# DataFrame augmentation helpers
# ---------------------------------------------------------------------------

def add_vcg_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add p_feasible, depth, num_gates columns to a VCG DataFrame."""
    df = df.copy()
    df['p_feasible'] = df.apply(p_feasible_vcg, axis=1)
    if 'resources' in df.columns:
        df['depth'] = df['resources'].apply(
            lambda r: r.depth if r is not None else float('nan')
        )
        df['num_gates'] = df['resources'].apply(
            lambda r: r.num_gates if r is not None else float('nan')
        )
    df['AR'] = pd.to_numeric(df['AR'], errors='coerce')
    return df


def add_hybrid_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add p_feasible and p_optimal columns to a HybridQAOA DataFrame."""
    df = df.copy()
    df['p_feasible'] = df.apply(p_feasible_hybrid, axis=1)
    df['p_optimal'] = df.apply(p_optimal_hybrid, axis=1)
    df['AR'] = pd.to_numeric(df['AR'], errors='coerce')
    return df


def summary_stats(df: pd.DataFrame, groupby: list, metrics: list) -> pd.DataFrame:
    """Compute per-group summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
    groupby : list of column names to group on
    metrics : list of metric column names to summarise

    Returns
    -------
    pd.DataFrame with columns: groupby cols + metric_{mean,std,median,q25,q75}
    """
    agg_funcs = {
        m: ['mean', 'std', 'median',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)]
        for m in metrics
    }
    summary = df.groupby(groupby)[metrics].agg(
        ['mean', 'std', 'median',
         lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75)]
    )
    # Flatten multi-level columns
    summary.columns = [
        f'{m}_{s}' if s not in ('<lambda_0>', '<lambda_1>')
        else f'{m}_{"q25" if s == "<lambda_0>" else "q75"}'
        for m, s in summary.columns
    ]
    return summary.reset_index()
