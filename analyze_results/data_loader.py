"""
data_loader.py -- Load and filter VCG / HybridQAOA result DataFrames.
"""

import glob as _glob
import os
import re
import pandas as pd


# Map filename substrings to canonical constraint-type labels
_FILENAME_PATTERNS = [
    (r'cardinality',        'cardinality'),
    (r'knapsack',           'knapsack'),
    (r'quadratic',          'quadratic'),
    (r'flow',               'flow'),
    (r'assignment',         'assignment'),
    (r'subtour',            'subtour'),
    (r'independent_set',    'independent_set'),
]


def infer_constraint_type(filepath: str) -> str:
    """Infer constraint family from filename using known patterns."""
    name = os.path.basename(filepath).lower()
    for pattern, label in _FILENAME_PATTERNS:
        if re.search(pattern, name):
            return label
    return 'unknown'


def _load_pickles(path_or_glob: str) -> pd.DataFrame:
    """Load one or more pickle files matching a path/glob into a DataFrame."""
    paths = sorted(_glob.glob(path_or_glob)) or [path_or_glob]
    frames = []
    for p in paths:
        df = pd.read_pickle(p)
        if 'constraint_type' not in df.columns or df['constraint_type'].eq('').all():
            df['constraint_type'] = infer_constraint_type(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_vcg_results(path_or_glob: str) -> pd.DataFrame:
    """Load VCG result pickle(s) into a DataFrame.

    Parameters
    ----------
    path_or_glob : str
        Path to a single pickle file, or a glob pattern matching several.

    Returns
    -------
    pd.DataFrame with ``constraint_type`` inferred from filename when absent.
    """
    return _load_pickles(path_or_glob)


def load_hybrid_results(path_or_glob: str) -> pd.DataFrame:
    """Load HybridQAOA result pickle(s) into a DataFrame."""
    return _load_pickles(path_or_glob)


def filter_df(df: pd.DataFrame, **criteria) -> pd.DataFrame:
    """Filter a DataFrame by equality on column values.

    Parameters
    ----------
    df : pd.DataFrame
    **criteria : keyword arguments mapping column name → value(s)
        A list value tests membership; a scalar tests equality.

    Examples
    --------
    >>> filter_df(df, n_x=3, angle_strategy='QAOA')
    >>> filter_df(df, constraint_type=['cardinality', 'knapsack'])
    """
    mask = pd.Series(True, index=df.index)
    for col, val in criteria.items():
        if col not in df.columns:
            continue
        if isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(val)
        else:
            mask &= df[col] == val
    return df[mask].reset_index(drop=True)
