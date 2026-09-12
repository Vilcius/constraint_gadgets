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
    ax.set_xticklabels([f'$n={n}$' for n in ns])
    ax.set_xlabel('$n$')
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
    """Side-by-side box + strip: AR distribution for PC-QAOA vs PenaltyQAOA."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df['method'].values]
    colors = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
    data = [df[df['method'] == m]['AR'].dropna().values for m in methods]

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
    ax.set_title('PC-QAOA vs PenaltyQAOA: AR')
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
        stds = grp.groupby('layer')['AR'].std().fillna(0)
        ax.plot(means.index, means.values, marker='o', label=method, color=color)
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.2, color=color)

    ax.set_xlabel('QAOA layers (p)')
    ax.set_ylabel('Mean AR')
    ax.set_title('AR vs QAOA layers')
    ax.legend(fontsize=10, framealpha=0.4)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_feas_vs_nx(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Line plot: mean AR_feas vs problem size (n_x), one line per method.

    P(feas)=0 instances contribute AR_feas=0 (worst case) rather than being
    excluded, so means reflect all experiments.
    """
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, grp in df.groupby('method'):
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['AR_feas'].mean()
        stds = grp.groupby('n_x')['AR_feas'].std().fillna(0)
        ax.plot(means.index, means.values, marker='o', label=method, color=color)
        ax.fill_between(means.index,
                        np.clip(means.values - stds.values, 0, None),
                        np.clip(means.values + stds.values, 0, 1),
                        alpha=0.2, color=color)

    ax.set_xlabel('Problem size ($n$)')
    ax.set_ylabel(r'Mean $\mathrm{AR}_{\mathrm{feas}}$')
    ax.set_title(r'$\mathrm{AR}_{\mathrm{feas}}$ vs problem size')
    ax.set_xticks(sorted(df['n_x'].unique()))
    ax.legend(fontsize=10, framealpha=0.4)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_ar_feas_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Side-by-side box + strip: AR_feas for PC-QAOA vs PenaltyQAOA.

    P(feas)=0 instances contribute AR_feas=0 (worst case) and are included.
    """
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df['method'].values]
    colors = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
    data = [df[df['method'] == m]['AR_feas'].values for m in methods]

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
    ax.set_title('AR$_\\mathrm{feas}$: PC-QAOA vs PenaltyQAOA\n'
                 '($P(\\mathrm{feas})=0$ instances assigned AR$_\\mathrm{feas}=0$)')
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

    # comp_ar_conv already has one row per experiment with n_layers = convergence layer
    last = df.copy()
    last['converged'] = last['p_feasible'] >= 0.75

    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in last['method'].values]
    colors = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
    max_layers = int(last['n_layers'].max())

    # x positions: 1..max_layers for converged, then immediately "Did not meet"
    x_converged = np.arange(1, max_layers + 1)
    x_dnm = max_layers + 1          # adjacent, separator drawn manually
    bar_width = 0.35

    for i, (method, color) in enumerate(zip(methods, colors)):
        sub = last[last['method'] == method]
        offset = (i - (len(methods) - 1) / 2) * bar_width

        # Converged bars
        counts = (sub[sub['converged']]['n_layers']
                  .value_counts()
                  .reindex(range(1, max_layers + 1), fill_value=0))
        bars = ax.bar(x_converged + offset, counts.values,
                      width=bar_width, color=color, edgecolor='white',
                      linewidth=0.7, alpha=0.85, zorder=3,
                      label=method)
        for bar, cnt in zip(bars, counts.values):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(cnt), ha='center', va='bottom', fontsize=7, zorder=4)

        # Did-not-meet bar: only tasks that ran to p_max and still didn't converge
        # (tasks with n_layers < p_max are still running — exclude them)
        dnm_count = ((~sub['converged']) & (sub['n_layers'] == max_layers)).sum()
        dnm_bar = ax.bar(x_dnm + offset, dnm_count,
                         width=bar_width, color=color, edgecolor='white',
                         linewidth=0.7, alpha=0.45, hatch='//', zorder=3)
        if dnm_count > 0:
            ax.text(dnm_bar[0].get_x() + dnm_bar[0].get_width() / 2,
                    dnm_count + 0.5, str(dnm_count),
                    ha='center', va='bottom', fontsize=7, zorder=4)

    # x-axis ticks
    all_x = list(x_converged) + [x_dnm]
    all_labels = [str(p) for p in range(1, max_layers + 1)] + ['Did not\nmeet']
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)

    # Vertical separator between layer bars and "Did not meet"
    ax.axvline(x=max_layers + 0.5, color=pu._ROSE_PINE['muted'],
               linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_xlabel('QAOA layers $p$ at convergence')
    ax.set_ylabel('Number of experiments')
    ax.set_title('Layers until $P(\\mathrm{feas}) \\geq 0.75$: PC-QAOA vs PenaltyQAOA')
    ax.legend(framealpha=1, fontsize=10)

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


def _plot_metric_panel(ax, df: pd.DataFrame, metric: str, ylabel: str,
                       group_col: str = 'method', linestyles: dict = None,
                       threshold: float | None = None) -> None:
    """Helper: plot mean ± std of *metric* vs n_x on *ax*, grouped by *group_col*."""
    nx_vals = sorted(df['n_x'].unique())
    for grp_val, grp in df.groupby(group_col):
        if group_col == 'method':
            color = pu.METHOD_COLORS.get(grp_val, pu._ROSE_PINE['subtle'])
            label = grp_val
        else:
            # overlap_type grouping
            color = pu._ROSE_PINE['pine'] if grp_val == 'disjoint' else pu._ROSE_PINE['iris']
            label = grp_val.capitalize()
        ls = (linestyles or {}).get(grp_val, '-')
        means = grp.groupby('n_x')[metric].mean()
        stds = grp.groupby('n_x')[metric].std().fillna(0)
        ax.plot(nx_vals, [means.get(n, np.nan) for n in nx_vals],
                marker='o', label=label, color=color, linestyle=ls)
        ax.fill_between(
            nx_vals,
            [max(0, means.get(n, 0) - stds.get(n, 0)) for n in nx_vals],
            [min(1, means.get(n, 0) + stds.get(n, 0)) for n in nx_vals],
            color=color, alpha=0.15,
        )
    if threshold is not None:
        ax.axhline(threshold, color=pu._ROSE_PINE['muted'],
                   linestyle='--', linewidth=1, label=f'threshold ({threshold})')
    ax.set_xlabel('$n$')
    ax.set_ylabel(ylabel)
    ax.set_xticks(nx_vals)
    ax.legend(fontsize=10, framealpha=0.4)


_OV_LS = {'disjoint': '-', 'overlapping': '--'}


def _split_panel_ar(ax, df: pd.DataFrame, metric: str, ylabel: str,
                    clip=(0.0, 1.0)) -> None:
    """4-line panel: both methods × {disjoint, overlapping}.

    Color = method, linestyle = overlap type (solid/dashed).
    """
    nx_vals = sorted(df['n_x'].unique())
    for method, mgrp in df.groupby('method'):
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        for ov, ogrp in mgrp.groupby('overlap_type'):
            ls = _OV_LS.get(ov, '-')
            label = f'{method} ({ov})'
            means = ogrp.groupby('n_x')[metric].mean()
            stds = ogrp.groupby('n_x')[metric].std().fillna(0)
            ax.plot(nx_vals, [means.get(n, np.nan) for n in nx_vals],
                    marker='o', linestyle=ls, label=label, color=color)
            ax.fill_between(
                nx_vals,
                [max(clip[0], means.get(n, 0) - stds.get(n, 0)) for n in nx_vals],
                [min(clip[1], means.get(n, 0) + stds.get(n, 0)) for n in nx_vals],
                color=color, alpha=0.10,
            )
    ax.set_xlabel('$n$')
    ax.set_ylabel(ylabel)
    ax.set_xticks(nx_vals)
    leg = ax.legend(fontsize=10, ncol=2, handlelength=2.5, framealpha=0.4)
    for h in leg.legend_handles:
        h.set_marker('')


def plot_ar_feas_with_overlap_split(
    df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """1x2: (a) AR_feas vs n_x both methods; (b) both methods by overlap type (4 lines)."""
    pu.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    _plot_metric_panel(axes[0], df, 'AR_feas',
                       r'Mean $\mathrm{AR}_\mathrm{feas}$')
    axes[0].set_title(r'$\mathrm{AR}_\mathrm{feas}$ vs $n$')

    _split_panel_ar(axes[1], df, 'AR_feas',
                    r'Mean $\mathrm{AR}_\mathrm{feas}$')
    axes[1].set_title(r'$\mathrm{AR}_\mathrm{feas}$: disjoint (—) vs overlapping (- -)')

    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig
