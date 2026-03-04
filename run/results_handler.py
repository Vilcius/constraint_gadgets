"""
results_handler.py -- ResultsCollector for accumulating experiment results.

Mirrors squbo's ResultsCollector pattern: append dicts incrementally,
resume from existing pickles, and export to DataFrame.
"""

import os
import pandas as pd


class ResultsCollector:
    """Accumulate per-experiment result dicts and persist to pickle.

    Usage
    -----
        collector = ResultsCollector()
        # optionally resume:  collector.load('results/my_run.pkl')
        collector.add({'n_x': 3, 'AR': 0.82, ...})
        collector.save('results/my_run.pkl')
        df = collector.to_dataframe()
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, row_dict: dict) -> None:
        """Append a single result dict (one experiment row)."""
        self._rows.append(row_dict)

    def load(self, path: str) -> None:
        """Resume from an existing pickle, extending any rows already held."""
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
