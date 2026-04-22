"""
results_helper.py -- Shared experiment utilities for constraint_gadget.

Provides:
    ResultsCollector                -- incremental result accumulation and pickle persistence
    read_typed_csv(filepath)        -- parse n_vars; ['c1', ...] CSV format
    remap_constraint_to_vars(...)   -- embed zero-indexed constraint into QUBO positions
"""

import os
import re

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# ResultsCollector
# ─────────────────────────────────────────────────────────────────────────────

class ResultsCollector:
    """Accumulate per-experiment result dicts and persist to pickle.

    Usage
    -----
        collector = ResultsCollector()
        collector.load('results/my_run.pkl')   # resume if exists
        collector.add(row_dict)                # append one experiment row
        collector.save('results/my_run.pkl')   # persist to pickle
        df = collector.to_dataframe()
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, row_dict: dict) -> None:
        """Append a single result dict (one experiment row)."""
        self._rows.append(row_dict)

    def load(self, path: str) -> None:
        """Resume from an existing pickle, prepending any rows already held."""
        if os.path.exists(path):
            existing = pd.read_pickle(path)
            if not existing.empty:
                self._rows = existing.to_dict('records') + self._rows

    def save(self, path: str) -> None:
        """Save all accumulated rows as a pandas pickle."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.to_dataframe().to_pickle(path)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all accumulated rows as a DataFrame."""
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)

# ─────────────────────────────────────────────────────────────────────────────
# Constraint string helpers
# ─────────────────────────────────────────────────────────────────────────────

def remap_constraint_to_vars(constraint_str: str, target_indices: list) -> str:
    """Embed a zero-indexed constraint into specific QUBO variable positions.

    Maps x_0 → x_{target_indices[0]}, x_1 → x_{target_indices[1]}, etc.
    Used to place a constraint on an arbitrary subset of a larger QUBO.

    Parameters
    ----------
    constraint_str : str
        Zero-indexed constraint, e.g. 'x_0 + x_1 + x_2 == 1'.
    target_indices : list[int]
        QUBO variable indices to assign, e.g. [2, 3, 4].

    Returns
    -------
    str
        Remapped constraint, e.g. 'x_2 + x_3 + x_4 == 1'.
    """
    def _replace(m):
        idx = int(m.group(1))
        return f'x_{target_indices[idx]}'
    return re.sub(r'x_(\d+)', _replace, constraint_str)


def read_typed_csv(filepath: str) -> list:
    """Parse a typed constraint CSV into a list of (n_vars, constraints) tuples.

    Expected line format:
        n_vars; ['constraint_string_1', 'constraint_string_2', ...]

    Returns
    -------
    list of (int, list[str])
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_str, rest = line.split('; ', 1)
            n_vars = int(n_str)
            parsed = eval(rest)
            if isinstance(parsed, str):
                constraints = [parsed]
            elif isinstance(parsed, tuple):
                constraints = list(parsed)
            else:
                constraints = list(parsed)
            rows.append((n_vars, constraints))
    return rows

