"""
statistical_tests.py -- Non-parametric statistical comparisons.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats


def compare_angle_strategies(df: pd.DataFrame,
                              metric: str = 'AR') -> pd.DataFrame:
    """Mann-Whitney U test: QAOA vs ma-QAOA on a given metric.

    Returns a DataFrame with the U statistic and p-value, grouped by
    constraint_type (if present).
    """
    results = []
    groups = df.groupby('constraint_type') if 'constraint_type' in df.columns \
        else [('all', df)]
    for name, grp in groups:
        a = grp[grp['angle_strategy'] == 'QAOA'][metric].dropna()
        b = grp[grp['angle_strategy'] == 'ma-QAOA'][metric].dropna()
        if len(a) < 2 or len(b) < 2:
            results.append({'constraint_type': name, 'U': np.nan, 'p_value': np.nan})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        results.append({'constraint_type': name, 'U': u, 'p_value': p})
    return pd.DataFrame(results)


def compare_constraint_types(df: pd.DataFrame,
                              metric: str = 'AR') -> pd.DataFrame:
    """Kruskal-Wallis test across constraint families.

    Returns a single-row DataFrame with H statistic and p-value.
    """
    if 'constraint_type' not in df.columns:
        return pd.DataFrame({'H': [np.nan], 'p_value': [np.nan]})
    groups = [
        grp[metric].dropna().values
        for _, grp in df.groupby('constraint_type')
        if len(grp[metric].dropna()) >= 2
    ]
    if len(groups) < 2:
        return pd.DataFrame({'H': [np.nan], 'p_value': [np.nan]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h, p = stats.kruskal(*groups)
    return pd.DataFrame({'H': [h], 'p_value': [p]})


def run_full_stats(vcg_df: pd.DataFrame, hybrid_df: pd.DataFrame,
                   output_dir: str = './analysis_output/statistical_tests/') -> None:
    """Run all statistical tests and export results as CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    # VCG angle strategy comparison
    if not vcg_df.empty and 'AR' in vcg_df.columns and 'angle_strategy' in vcg_df.columns:
        angle_vcg = compare_angle_strategies(vcg_df, 'AR')
        angle_vcg.to_csv(os.path.join(output_dir, 'vcg_angle_strategy_mannwhitney.csv'),
                         index=False)
        kw_vcg = compare_constraint_types(vcg_df, 'AR')
        kw_vcg.to_csv(os.path.join(output_dir, 'vcg_constraint_type_kruskalwallis.csv'),
                      index=False)

    # Hybrid AR comparison across constraint types
    if not hybrid_df.empty and 'AR' in hybrid_df.columns:
        kw_hybrid = compare_constraint_types(hybrid_df, 'AR')
        kw_hybrid.to_csv(os.path.join(output_dir, 'hybrid_constraint_type_kruskalwallis.csv'),
                         index=False)

    print(f"Statistical test results saved to {output_dir}")
