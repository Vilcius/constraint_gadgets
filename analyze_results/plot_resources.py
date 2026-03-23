"""
plot_resources.py -- Circuit resource and timing plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_utils as pu


def plot_shots_vs_n(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Estimated shots vs n_x (scatter + mean line)."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for family, grp in df.groupby('constraint_type'):
        color = pu.CONSTRAINT_COLORS.get(family, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['est_shots'].mean()
        ax.plot(means.index, means.values, marker='o', label=family, color=color)
        ax.scatter(grp['n_x'], grp['est_shots'], color=color, alpha=0.3, s=20)

    ax.set_xlabel('n_x')
    ax.set_ylabel('Estimated shots')
    ax.set_title('Estimated shots vs n_x')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_depth_vs_n(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Circuit depth vs n_x."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if 'depth' not in df.columns:
        ax.text(0.5, 0.5, 'No depth data', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    for family, grp in df.groupby('constraint_type'):
        color = pu.CONSTRAINT_COLORS.get(family, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['depth'].mean()
        ax.plot(means.index, means.values, marker='s', label=family, color=color)
        ax.scatter(grp['n_x'], grp['depth'], color=color, alpha=0.3, s=20)

    ax.set_xlabel('n_x')
    ax.set_ylabel('Circuit depth')
    ax.set_title('Circuit depth vs n_x')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def add_total_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``total_time`` column (sum of component times) if not already present.

    Works on both old results (no total_time column) and new results (already has it).
    total_time per row = optimize_time + counts_time + hamiltonian_time (if present).
    This is the wall-clock time for a *single* QAOA layer.

    To get the full solve time across all layers for a task, sum total_time
    grouped by (qubo_string, method) or similar task identifier.
    """
    if 'total_time' in df.columns:
        return df
    df = df.copy()

    def _scalar(v):
        return float(v[0]) if isinstance(v, list) else float(v) if v is not None else 0.0

    opt  = df['optimize_time'].apply(_scalar) if 'optimize_time' in df.columns else 0.0
    cnt  = df['counts_time'].apply(_scalar)   if 'counts_time'   in df.columns else 0.0
    ham  = df['hamiltonian_time'].apply(_scalar) if 'hamiltonian_time' in df.columns else 0.0
    df['total_time'] = opt + cnt + ham
    return df


def plot_time_breakdown(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Stacked bar: mean hamiltonian / optimize / counts time per n_x."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    time_cols = ['hamiltonian_time', 'optimize_time', 'counts_time']
    present = [c for c in time_cols if c in df.columns]
    if not present:
        ax.text(0.5, 0.5, 'No timing data', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    df_plot = df.copy()
    for col in present:
        df_plot[col] = df_plot[col].apply(
            lambda v: v[0] if isinstance(v, list) else v).astype(float)
    means = df_plot.groupby('n_x')[present].mean()
    ns = means.index.tolist()
    x = np.arange(len(ns))
    width = 0.6
    colors = [pu._ROSE_PINE['pine'], pu._ROSE_PINE['iris'], pu._ROSE_PINE['gold']]
    labels = ['Hamiltonian', 'Optimize', 'Counts']

    bottoms = np.zeros(len(ns))
    for col, color, label in zip(present, colors, labels):
        vals = means[col].fillna(0).values
        ax.bar(x, vals, width, bottom=bottoms, color=color, alpha=0.8, label=label)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('n_x')
    ax.set_ylabel('Time (s)')
    ax.set_title('Mean time breakdown by n_x')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_total_time_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Box + strip: cumulative solve time per task, HybridQAOA vs PenaltyQAOA.

    ``df`` must contain a ``method`` column with values 'HybridQAOA' / 'PenaltyQAOA'
    and a task identifier (``qubo_string``) so that per-layer times can be summed
    into a single total-solve-time per task.

    Per-layer ``total_time`` is computed via ``add_total_time`` if not present.
    """
    if 'method' not in df.columns:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No method column', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    pu.setup_style()
    df = add_total_time(df)

    def _scalar(v):
        return float(v[0]) if isinstance(v, list) else float(v) if v is not None else 0.0

    df = df.copy()
    df['total_time'] = df['total_time'].apply(_scalar)

    task_key = 'qubo_string' if 'qubo_string' in df.columns else df.index.name or df.index

    # Prefer cumulative_time (last layer value = total solve time) when available,
    # otherwise sum total_time across layers per task.
    if 'cumulative_time' in df.columns:
        df['cumulative_time'] = df['cumulative_time'].apply(_scalar)
        cumulative = (
            df.groupby([task_key, 'method'])['cumulative_time']
            .max()
            .reset_index()
            .rename(columns={'cumulative_time': 'solve_time'})
        )
    else:
        cumulative = (
            df.groupby([task_key, 'method'])['total_time']
            .sum()
            .reset_index()
            .rename(columns={'total_time': 'solve_time'})
        )

    methods = ['HybridQAOA', 'PenaltyQAOA']
    colors  = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(42)
    positions = list(range(len(methods)))

    data_by_method = [
        cumulative[cumulative['method'] == m]['solve_time'].values
        for m in methods
    ]

    bp = ax.boxplot(data_by_method, positions=positions, patch_artist=True,
                    widths=0.4,
                    medianprops=dict(color=pu._ROSE_PINE['text'], linewidth=2),
                    whiskerprops=dict(color=pu._ROSE_PINE['muted']),
                    capprops=dict(color=pu._ROSE_PINE['muted']),
                    flierprops=dict(marker='', alpha=0),
                    zorder=2)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    for xi, (m, color, vals) in enumerate(zip(methods, colors, data_by_method)):
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=25, alpha=0.7, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Total solve time (s)')
    ax.set_title('Cumulative solve time per task\n(sum across all layers run)')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig
