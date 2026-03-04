"""
plot_ar.py -- Approximation-ratio (AR) plots.
"""

import matplotlib.pyplot as plt
import pandas as pd

from . import plot_utils as pu


def plot_ar_by_n(df: pd.DataFrame, title: str = 'AR vs n_x',
                 save_path: str = None) -> plt.Figure:
    """Box plot of AR vs number of decision variables."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ns = sorted(df['n_x'].unique())
    data = [df[df['n_x'] == n]['AR'].dropna().values for n in ns]

    bp = ax.boxplot(data, patch_artist=True, medianprops={'color': pu._ROSE_PINE['gold']})
    color = pu._ROSE_PINE['pine']
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(ns) + 1))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('n_x')
    ax.set_ylabel('Approximation Ratio (AR)')
    ax.set_title(title)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_by_constraint_type(vcg_df: pd.DataFrame,
                                save_path: str = None) -> plt.Figure:
    """Box plot of AR across constraint families."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    families = sorted(vcg_df['constraint_type'].unique())
    data = [vcg_df[vcg_df['constraint_type'] == f]['AR'].dropna().values
            for f in families]
    colors = [pu.CONSTRAINT_COLORS.get(f, pu._ROSE_PINE['subtle']) for f in families]

    bp = ax.boxplot(data, patch_artist=True, medianprops={'color': pu._ROSE_PINE['gold']})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(families) + 1))
    ax.set_xticklabels(families, rotation=20, ha='right')
    ax.set_xlabel('Constraint type')
    ax.set_ylabel('Approximation Ratio (AR)')
    ax.set_title('AR by constraint family')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_by_angle_strategy(df: pd.DataFrame,
                               save_path: str = None) -> plt.Figure:
    """Side-by-side box plots of AR for QAOA vs ma-QAOA."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    strategies = sorted(df['angle_strategy'].unique())
    data = [df[df['angle_strategy'] == s]['AR'].dropna().values for s in strategies]
    colors = [pu.ANGLE_COLORS.get(s, pu._ROSE_PINE['subtle']) for s in strategies]

    bp = ax.boxplot(data, patch_artist=True, medianprops={'color': pu._ROSE_PINE['gold']})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(strategies) + 1))
    ax.set_xticklabels(strategies)
    ax.set_xlabel('Angle strategy')
    ax.set_ylabel('Approximation Ratio (AR)')
    ax.set_title('AR: QAOA vs ma-QAOA')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig
