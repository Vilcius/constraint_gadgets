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


def plot_ar_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Scatter: HybridQAOA AR vs PenaltyQAOA AR, one point per (task, layer)."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    hybrid  = df[df['method'] == 'HybridQAOA'].set_index(['constraint_type', 'n_x', 'layer'])
    penalty = df[df['method'] == 'PenaltyQAOA'].set_index(['constraint_type', 'n_x', 'layer'])
    common  = hybrid.index.intersection(penalty.index)

    if common.empty:
        ax.text(0.5, 0.5, 'No matched (task, layer) pairs', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    h_ar = hybrid.loc[common, 'AR'].values
    p_ar = penalty.loc[common, 'AR'].values

    ax.scatter(p_ar, h_ar, alpha=0.5, s=30, color=pu._ROSE_PINE['pine'])
    lims = [min(h_ar.min(), p_ar.min()) - 0.02, max(h_ar.max(), p_ar.max()) + 0.02]
    ax.plot(lims, lims, '--', color=pu._ROSE_PINE['subtle'], linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('PenaltyQAOA AR')
    ax.set_ylabel('HybridQAOA AR')
    ax.set_title('HybridQAOA vs PenaltyQAOA: AR')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_vs_layers(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Line plot: mean AR vs QAOA layer, one line per method."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, grp in df.groupby('method'):
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        means = grp.groupby('layer')['AR'].mean()
        stds  = grp.groupby('layer')['AR'].std().fillna(0)
        ax.plot(means.index, means.values, marker='o', label=method, color=color)
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.2, color=color)

    ax.set_xlabel('QAOA layers (p)')
    ax.set_ylabel('Mean AR')
    ax.set_title('AR vs QAOA layers')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_feas_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Side-by-side scatter + box: AR_feas for HybridQAOA vs PenaltyQAOA.

    Left panel: scatter AR_feas(Hybrid) vs AR_feas(Penalty), paired by task+layer.
    Right panel: box plot of AR_feas distribution per method.
    Undefined (NaN) rows are silently excluded (P(feas)=0 cases).
    """
    pu.setup_style()
    fig, (ax_sc, ax_bx) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('AR_feas  (feasibility-conditioned approximation ratio)\n'
                 'Undefined when P(feas)=0; excludes infeasible shots from denominator',
                 fontsize=10)

    C = pu._ROSE_PINE

    # --- scatter ---
    hybrid  = df[df['method'] == 'HybridQAOA'].dropna(subset=['AR_feas'])
    penalty = df[df['method'] == 'PenaltyQAOA'].dropna(subset=['AR_feas'])

    h_idx = hybrid.set_index(['constraint_type', 'n_x', 'layer'])
    p_idx = penalty.set_index(['constraint_type', 'n_x', 'layer'])
    common = h_idx.index.intersection(p_idx.index)

    if not common.empty:
        h_vals = h_idx.loc[common, 'AR_feas'].values
        p_vals = p_idx.loc[common, 'AR_feas'].values
        ax_sc.scatter(p_vals, h_vals, alpha=0.5, s=30, color=C['pine'])
        lims = [min(h_vals.min(), p_vals.min()) - 0.02,
                max(h_vals.max(), p_vals.max()) + 0.02]
        ax_sc.plot(lims, lims, '--', color=C['subtle'], linewidth=1)
        ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
    else:
        ax_sc.text(0.5, 0.5, 'No paired rows', transform=ax_sc.transAxes,
                   ha='center', va='center')

    ax_sc.set_xlabel('PenaltyQAOA  AR_feas')
    ax_sc.set_ylabel('HybridQAOA  AR_feas')
    ax_sc.set_title('AR_feas: HybridQAOA vs PenaltyQAOA')

    # --- box ---
    methods = [m for m in ['HybridQAOA', 'PenaltyQAOA']
               if m in df['method'].values]
    data   = [df[df['method'] == m]['AR_feas'].dropna().values for m in methods]
    colors = [pu.METHOD_COLORS.get(m, C['subtle']) for m in methods]

    if any(len(d) > 0 for d in data):
        bp = ax_bx.boxplot(data, patch_artist=True,
                           medianprops={'color': C['gold']})
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax_bx.set_xticks(range(1, len(methods) + 1))
    ax_bx.set_xticklabels(methods)
    ax_bx.set_ylabel('AR_feas')
    ax_bx.set_title('AR_feas distribution by method')

    plt.tight_layout()
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
