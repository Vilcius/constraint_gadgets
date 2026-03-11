"""
metrics.py -- Core metric functions for VCG and HybridQAOA results.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Raw-counts helpers (used by examples and DataFrame augmentation)
# ---------------------------------------------------------------------------

def aggregate_counts(counts: dict, n_x: int) -> dict:
    """Collapse auxiliary bits; return {decision_bitstring: probability}.

    Parameters
    ----------
    counts : dict mapping full bitstring -> shot count
    n_x : number of decision-variable bits (prefix to keep)
    """
    total = sum(counts.values())
    agg: dict = {}
    for bs, cnt in counts.items():
        key = bs[:n_x]
        agg[key] = agg.get(key, 0) + cnt / total
    return agg


def feasibility_check(bitstring: str, constraints: list, n_x: int) -> bool:
    """Return True if the first n_x bits of bitstring satisfy all constraints.

    Parameters
    ----------
    bitstring : str of '0'/'1' characters (may include auxiliary bits)
    constraints : list of constraint strings, e.g. ['x_0 + x_1 == 1']
    n_x : number of decision variables to read from the prefix
    """
    vd = {f'x_{i}': int(bitstring[i]) for i in range(n_x)}
    return all(eval(c, {"__builtins__": {}}, vd) for c in constraints)


def compute_comparison_metrics(counts: dict, opt_cost: float,
                                C_max: float, C_min: float,
                                constraints: list, n_x: int,
                                optimal_x: list = None) -> dict:
    """Compute AR, P(feasible), P(optimal) from raw measurement counts.

    Parameters
    ----------
    counts : dict mapping bitstring -> shot count
    opt_cost : optimised expectation value ⟨H⟩
    C_max, C_min : max/min eigenvalues of the cost Hamiltonian
    constraints : list of constraint strings (decision-variable indexed)
    n_x : number of decision variables
    optimal_x : list of optimal decision bitstrings, or None

    Returns
    -------
    dict with keys AR, p_feasible, p_optimal
    """
    agg = aggregate_counts(counts, n_x)
    p_feas = sum(p for bs, p in agg.items()
                 if feasibility_check(bs, constraints, n_x))
    if optimal_x:
        p_opt = sum(p for bs, p in agg.items() if bs in optimal_x)
    else:
        p_opt = float('nan')
    ar = (float(opt_cost) - C_max) / (C_min - C_max)
    return dict(AR=ar, p_feasible=p_feas, p_optimal=p_opt)


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
    Accepts both raw single-row dicts (values list-wrapped) and scalar-valued rows.
    """
    counts = row['counts']
    if isinstance(counts, list):
        counts = counts[0]
    n_x = row['n_x']
    if isinstance(n_x, list):
        n_x = n_x[0]
    n_x = int(n_x)
    constraints = row['constraints']
    if isinstance(constraints, list) and constraints and isinstance(constraints[0], list):
        constraints = constraints[0]
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
    """Fraction of counts (first n_x bits) that are an optimal feasible solution.

    Uses the ``optimal_x`` list of bitstrings stored by the run script (produced
    by ``data.make_data.get_optimal_x`` via brute-force enumeration).
    Returns NaN if ``optimal_x`` is absent or None (e.g. no feasible solution
    exists, or legacy results recorded before the field was added).
    """
    optimal_x = row.get('optimal_x')
    if isinstance(optimal_x, list) and optimal_x and isinstance(optimal_x[0], list):
        optimal_x = optimal_x[0]
    if not optimal_x:
        return float('nan')

    counts = row['counts']
    if isinstance(counts, list):
        counts = counts[0]
    n_x = row['n_x']
    if isinstance(n_x, list):
        n_x = n_x[0]
    n_x = int(n_x)

    total = sum(counts.values())
    if total == 0:
        return float('nan')

    optimal_set = set(optimal_x)
    optimal = sum(cnt for bs, cnt in counts.items() if bs[:n_x] in optimal_set)
    return optimal / total


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
