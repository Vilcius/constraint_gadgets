"""
plot_ar.py -- Approximation-ratio (AR) plots.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from . import plot_utils as pu


def plot_ar_by_n(df: pd.DataFrame, title: str = 'AR vs n_x',
                 save_path: str = None) -> plt.Figure:
    """Box plot of AR vs number of decision variables."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ns = sorted(df['n_x'].unique())
    data = [df[df['n_x'] == n]['AR'].dropna().values for n in ns]

    rng = np.random.default_rng(42)
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color=pu._ROSE_PINE['text'], linewidth=2),
                    whiskerprops=dict(color=pu._ROSE_PINE['muted']),
                    capprops=dict(color=pu._ROSE_PINE['muted']),
                    flierprops=dict(marker='', alpha=0),
                    zorder=2)
    color = pu._ROSE_PINE['pine']
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for xi, vals in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + 1 + jitter, vals, color=color, s=25, alpha=0.75, zorder=3)

    ax.set_xticks(range(1, len(ns) + 1))
    ax.set_xticklabels([f'$n_x={n}$' for n in ns])
    ax.set_xlabel('$n_x$')
    ax.set_ylabel('Approximation Ratio (AR)')
    ax.set_title(title)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

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

    rng = np.random.default_rng(42)
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color=pu._ROSE_PINE['text'], linewidth=2),
                    whiskerprops=dict(color=pu._ROSE_PINE['muted']),
                    capprops=dict(color=pu._ROSE_PINE['muted']),
                    flierprops=dict(marker='', alpha=0),
                    zorder=2)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for xi, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + 1 + jitter, vals, color=color, s=25, alpha=0.75, zorder=3)

    ax.set_xticks(range(1, len(families) + 1))
    ax.set_xticklabels(families, rotation=20, ha='right')
    ax.set_xlabel('Constraint type')
    ax.set_ylabel('Approximation Ratio (AR)')
    ax.set_title('AR by constraint family')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Side-by-side box + strip: AR distribution for HybridQAOA vs PenaltyQAOA."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    methods = [m for m in ['HybridQAOA', 'PenaltyQAOA'] if m in df['method'].values]
    colors  = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
    data    = [df[df['method'] == m]['AR'].dropna().values for m in methods]

    positions = list(range(len(methods)))
    rng = np.random.default_rng(42)

    bp = ax.boxplot(data, positions=positions, patch_artist=True,
                    widths=0.4,
                    medianprops=dict(color=pu._ROSE_PINE['text'], linewidth=2),
                    whiskerprops=dict(color=pu._ROSE_PINE['muted']),
                    capprops=dict(color=pu._ROSE_PINE['muted']),
                    flierprops=dict(marker='', alpha=0),
                    zorder=2)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for xi, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=25, alpha=0.75, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Approximation Ratio (AR)')
    ax.set_title('HybridQAOA vs PenaltyQAOA: AR')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

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


def plot_ar_feas_vs_layers(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Line plot: mean AR_feas vs QAOA layer, one line per method.

    Rows with NaN AR_feas (P(feas)=0) are excluded before aggregating.
    """
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, grp in df.groupby('method'):
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        valid = grp.dropna(subset=['AR_feas'])
        means = valid.groupby('layer')['AR_feas'].mean()
        stds  = valid.groupby('layer')['AR_feas'].std().fillna(0)
        ax.plot(means.index, means.values, marker='o', label=method, color=color)
        ax.fill_between(means.index,
                        np.clip(means.values - stds.values, 0, None),
                        np.clip(means.values + stds.values, 0, 1),
                        alpha=0.2, color=color)

    ax.set_xlabel('QAOA layers ($p$)')
    ax.set_ylabel(r'Mean $\mathrm{AR}_{\mathrm{feas}}$')
    ax.set_title(r'$\mathrm{AR}_{\mathrm{feas}}$ vs QAOA layers')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_feas_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Side-by-side box + strip: AR_feas for HybridQAOA vs PenaltyQAOA.

    Undefined (NaN) rows are silently excluded (P(feas)=0 cases).
    """
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    methods = [m for m in ['HybridQAOA', 'PenaltyQAOA'] if m in df['method'].values]
    colors  = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
    data    = [df[df['method'] == m]['AR_feas'].dropna().values for m in methods]

    positions = list(range(len(methods)))
    rng = np.random.default_rng(42)

    if any(len(d) > 0 for d in data):
        bp = ax.boxplot(data, positions=positions, patch_artist=True,
                        widths=0.4,
                        medianprops=dict(color=pu._ROSE_PINE['text'], linewidth=2),
                        whiskerprops=dict(color=pu._ROSE_PINE['muted']),
                        capprops=dict(color=pu._ROSE_PINE['muted']),
                        flierprops=dict(marker='', alpha=0),
                        zorder=2)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

    for xi, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=25, alpha=0.75, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel('AR$_\\mathrm{feas}$')
    ax.set_title('AR$_\\mathrm{feas}$: HybridQAOA vs PenaltyQAOA\n'
                 '(undefined when $P(\\mathrm{feas})=0$)')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_layers_to_threshold(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Grouped bar: layers until P(feas)>=0.75, plus a 'Did not meet' bar.

    Each experiment contributes one count. Converged = final p_feasible >= 0.75;
    otherwise counted as 'Did not meet threshold'.
    """
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    last = df[df['layer'] == df['n_layers']].copy()
    last['converged'] = last['p_feasible'] >= 0.75

    methods    = [m for m in ['HybridQAOA', 'PenaltyQAOA'] if m in last['method'].values]
    colors     = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
    max_layers = int(last['n_layers'].max())

    # x positions: 1..max_layers for converged, then a gap, then "Did not meet"
    x_converged = np.arange(1, max_layers + 1)
    x_dnm       = max_layers + 2          # extra tick with a gap
    bar_width   = 0.35

    for i, (method, color) in enumerate(zip(methods, colors)):
        sub    = last[last['method'] == method]
        offset = (i - (len(methods) - 1) / 2) * bar_width

        # Converged bars
        counts = (sub[sub['converged']]['n_layers']
                  .value_counts()
                  .reindex(range(1, max_layers + 1), fill_value=0))
        ax.bar(x_converged + offset, counts.values,
               width=bar_width, color=color, edgecolor='white',
               linewidth=0.7, alpha=0.85, zorder=3,
               label=method)

        # Did-not-meet bar: only tasks that ran to p_max and still didn't converge
        # (tasks with n_layers < p_max are still running — exclude them)
        dnm_count = ((~sub['converged']) & (sub['n_layers'] == max_layers)).sum()
        ax.bar(x_dnm + offset, dnm_count,
               width=bar_width, color=color, edgecolor='white',
               linewidth=0.7, alpha=0.45, hatch='//', zorder=3)

    # x-axis ticks
    all_x      = list(x_converged) + [x_dnm]
    all_labels = [str(p) for p in range(1, max_layers + 1)] + ['Did not\nmeet']
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)

    # Vertical separator before "Did not meet"
    ax.axvline(x=max_layers + 1, color=pu._ROSE_PINE['muted'],
               linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_xlabel('QAOA layers $p$ at convergence')
    ax.set_ylabel('Number of experiments')
    ax.set_title('Layers until $P(\\mathrm{feas}) \\geq 0.75$: HybridQAOA vs PenaltyQAOA')
    ax.legend(framealpha=1)

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
