"""
plot_feasibility.py -- P(feasible) and P(optimal) plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from . import plot_utils as pu
from .metrics import aggregate_counts, feasibility_check


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


def plot_vcg_counts(rows: list, constraint_label: str = '',
                    save_path: str = None) -> plt.Figure:
    """Bar chart of VCG measurement distributions, one panel per angle strategy.

    Parameters
    ----------
    rows : list of result dicts from collect_vcg_data (one per angle strategy)
    constraint_label : displayed in the figure title
    save_path : if given, save figure to this path
    """
    pu.setup_style()
    colors = [pu.ANGLE_COLORS.get(r['angle_strategy'][0]
                                   if isinstance(r['angle_strategy'], list)
                                   else r['angle_strategy'],
                                   pu._ROSE_PINE['subtle'])
              for r in rows]
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4))
    if len(rows) == 1:
        axes = [axes]

    for ax, row, color in zip(axes, rows, colors):
        strategy = row['angle_strategy'][0] if isinstance(row['angle_strategy'], list) else row['angle_strategy']
        counts = row['counts'][0] if isinstance(row['counts'], list) else row['counts']
        keys = sorted(counts.keys())
        total = sum(counts.values())
        probs = [counts[k] / total for k in keys]
        ar = row['AR'][0] if isinstance(row['AR'], list) else row['AR']
        p_feas = sum(p for k, p in zip(keys, probs) if k[-1] == '0')

        ax.bar(keys, probs, color=color, alpha=0.85)
        ax.set_title(f'{strategy}  |  AR={ar:.3f}  P(feas)={p_feas:.3f}')
        ax.set_xlabel('Bitstring')
        ax.set_ylabel('Probability')
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(f'VCG measurement distribution\n{constraint_label}')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_method_comparison(metrics: dict, title: str = 'HybridQAOA vs PenaltyQAOA',
                           save_path: str = None) -> plt.Figure:
    """Grouped bar chart comparing methods on AR, P(feasible), P(optimal).

    Parameters
    ----------
    metrics : dict mapping method name -> dict with keys AR, p_feasible, p_optimal
              e.g. {'HybridQAOA': {...}, 'PenaltyQAOA': {...}}
    title : plot title
    save_path : if given, save figure to this path
    """
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    metric_keys = ['AR', 'p_feasible', 'p_optimal']
    labels = ['Approximation Ratio', 'P(feasible)', 'P(optimal)']
    x = np.arange(len(metric_keys))
    n_methods = len(metrics)
    width = 0.7 / n_methods

    method_names = list(metrics.keys())
    for i, name in enumerate(method_names):
        vals = [metrics[name][k] for k in metric_keys]
        color = pu.METHOD_COLORS.get(name, pu._ROSE_PINE['subtle'])
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_outcome_distributions(counts: dict, constraints: list, n_x: int,
                                optimal_x: list = None, top_n: int = 20,
                                title: str = 'Measurement distributions',
                                save_path: str = None) -> plt.Figure:
    """Side-by-side bar charts of top-N measurement outcomes, coloured by status.

    Parameters
    ----------
    counts : dict mapping method name -> {bitstring: shot_count}
    constraints : list of constraint strings (decision-variable indexed)
    n_x : number of decision variables
    optimal_x : list of optimal decision bitstrings for colour coding
    top_n : number of top outcomes to show per method
    title : overall figure title
    save_path : if given, save figure to this path
    """
    pu.setup_style()
    method_names = list(counts.keys())
    fig, axes = plt.subplots(1, len(method_names),
                             figsize=(7 * len(method_names), 5), sharey=False)
    if len(method_names) == 1:
        axes = [axes]

    def _bar_color(bs):
        if optimal_x and bs in optimal_x:
            return pu._ROSE_PINE['foam']
        if feasibility_check(bs, constraints, n_x):
            return pu._ROSE_PINE['pine']
        return pu._ROSE_PINE['love']

    for ax, name in zip(axes, method_names):
        agg = aggregate_counts(counts[name], n_x)
        top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        bstrings = [bs for bs, _ in top]
        probs = [p for _, p in top]
        colors = [_bar_color(bs) for bs in bstrings]

        ax.bar(range(len(bstrings)), probs, color=colors, alpha=0.85)
        ax.set_xticks(range(len(bstrings)))
        ax.set_xticklabels(bstrings, rotation=90, fontsize=7)
        ax.set_xlabel(f'Bitstring (decision variables x_0..x_{n_x - 1})')
        ax.set_ylabel('Probability')
        ax.set_title(f'{name} – top {top_n} outcomes')

        p_f = sum(p for bs, p in top if feasibility_check(bs, constraints, n_x))
        ax.text(0.98, 0.97, f'P(feas) shown: {p_f:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)

    legend_patches = [
        mpatches.Patch(color=pu._ROSE_PINE['foam'], label='Optimal'),
        mpatches.Patch(color=pu._ROSE_PINE['pine'], label='Feasible'),
        mpatches.Patch(color=pu._ROSE_PINE['love'], label='Infeasible'),
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.05)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig
