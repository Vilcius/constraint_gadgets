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
