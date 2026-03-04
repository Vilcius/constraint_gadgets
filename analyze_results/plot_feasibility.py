"""
plot_feasibility.py -- P(feasible) and P(optimal) plots.
"""

import matplotlib.pyplot as plt
import pandas as pd

from . import plot_utils as pu


def plot_p_feasible_vcg(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """P(feasible) vs n_x, coloured by constraint family."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for family, grp in df.groupby('constraint_type'):
        color = pu.CONSTRAINT_COLORS.get(family, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['p_feasible'].mean()
        stds  = grp.groupby('n_x')['p_feasible'].std().fillna(0)
        ax.plot(means.index, means.values, marker='o', label=family, color=color)
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.2, color=color)

    ax.set_xlabel('n_x')
    ax.set_ylabel('P(feasible)')
    ax.set_title('VCG: P(feasible) vs n_x')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_p_feasible_hybrid(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """P(feasible) vs n_x for HybridQAOA."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for family, grp in df.groupby('constraint_type'):
        color = pu.CONSTRAINT_COLORS.get(family, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['p_feasible'].mean()
        stds  = grp.groupby('n_x')['p_feasible'].std().fillna(0)
        ax.plot(means.index, means.values, marker='s', label=family, color=color)
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.2, color=color)

    ax.set_xlabel('n_x')
    ax.set_ylabel('P(feasible)')
    ax.set_title('HybridQAOA: P(feasible) vs n_x')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_p_optimal_hybrid(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """P(optimal) vs n_x for HybridQAOA."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for family, grp in df.groupby('constraint_type'):
        color = pu.CONSTRAINT_COLORS.get(family, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['p_optimal'].mean()
        stds  = grp.groupby('n_x')['p_optimal'].std().fillna(0)
        ax.plot(means.index, means.values, marker='^', label=family, color=color)
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.2, color=color)

    ax.set_xlabel('n_x')
    ax.set_ylabel('P(optimal)')
    ax.set_title('HybridQAOA: P(optimal) vs n_x')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_vs_p_feasible(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Scatter: AR vs P(feasible)."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    for family, grp in df.groupby('constraint_type'):
        color = pu.CONSTRAINT_COLORS.get(family, pu._ROSE_PINE['subtle'])
        ax.scatter(grp['p_feasible'], grp['AR'], label=family,
                   color=color, alpha=0.6, s=40)

    ax.set_xlabel('P(feasible)')
    ax.set_ylabel('AR')
    ax.set_title('AR vs P(feasible)')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig
