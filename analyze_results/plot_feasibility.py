"""
plot_feasibility.py -- P(feasible) and P(optimal) plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from . import plot_utils as pu
from .metrics import aggregate_counts, feasibility_check
from core import constraint_handler as ch


def plot_p_feasible_pcqaoa(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """P(feasible) vs n_x for PC-QAOA."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for family, grp in df.groupby('constraint_type'):
        color = pu.CONSTRAINT_COLORS.get(family, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['p_feasible'].mean()
        stds = grp.groupby('n_x')['p_feasible'].std().fillna(0)
        ax.plot(means.index, means.values, marker='s', label=family, color=color)
        ax.fill_between(means.index,
                        np.clip(means.values - stds.values, 0, 1),
                        np.clip(means.values + stds.values, 0, 1),
                        alpha=0.2, color=color)

    ax.set_xlabel('$n$')
    ax.set_ylabel('$P(\\mathrm{feas})$')
    ax.set_title('PC-QAOA: P(feasible) vs $n$')
    ax.legend(fontsize=10, framealpha=0.4)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_p_optimal_pcqaoa(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """P(optimal) vs n_x for PC-QAOA, aggregated over all constraint types."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate across all constraint-type combinations — too many to plot individually
    for method, grp in df.groupby('method'):
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['p_optimal'].mean()
        stds = grp.groupby('n_x')['p_optimal'].std().fillna(0)
        ax.plot(means.index, means.values, marker='o', label=method, color=color)
        ax.fill_between(means.index,
                        np.clip(means.values - stds.values, 0, None),
                        np.clip(means.values + stds.values, 0, 1),
                        alpha=0.2, color=color)

    ax.set_xlabel('$n$')
    ax.set_ylabel('P(optimal)')
    ax.set_title('P(optimal) vs $n$')
    ax.legend(fontsize=10, framealpha=0.4)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_p_feasible_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Line plot: mean P(feasible) vs n_x, one line per method."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, grp in df.groupby('method'):
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        means = grp.groupby('n_x')['p_feasible'].mean()
        stds = grp.groupby('n_x')['p_feasible'].std().fillna(0)
        ax.plot(means.index, means.values, marker='o', label=method, color=color)
        ax.fill_between(means.index,
                        np.clip(means.values - stds.values, 0, 1),
                        np.clip(means.values + stds.values, 0, 1),
                        alpha=0.2, color=color)

    ax.set_xlabel('$n$')
    ax.set_ylabel('$P(\\mathrm{feas})$')
    ax.set_title('$P(\\mathrm{feas})$ vs $n$: PC-QAOA vs PenaltyQAOA')
    ax.legend(fontsize=10, framealpha=0.4)

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
        constraints = row['constraints'][0] if isinstance(row['constraints'], list) else row['constraints']
        keys = sorted(counts.keys())
        total = sum(counts.values())
        probs = [counts[k] / total for k in keys]
        ar = row['AR'][0] if isinstance(row['AR'], list) else row['AR']
        parsed = ch.parse_constraints(constraints)
        is_feas = [ch.check_feasibility(k, parsed) for k in keys]
        p_feas = sum(p for p, f in zip(probs, is_feas) if f)
        bar_colors = [color if f else pu._ROSE_PINE.get('love', 'salmon') for f in is_feas]

        ax.bar(keys, probs, color=bar_colors, alpha=0.85)
        ax.set_title(f'{strategy}  |  AR={ar:.3f}  P(feas)={p_feas:.3f}')
        ax.set_xlabel('Bitstring')
        ax.set_ylabel('Probability')
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(f'VCG measurement distribution\n{constraint_label}')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_method_comparison(metrics: dict, title: str = 'PC-QAOA vs PenaltyQAOA',
                           save_path: str = None) -> plt.Figure:
    """Grouped bar chart comparing methods on AR, P(feasible), P(optimal).

    Parameters
    ----------
    metrics : dict mapping method name -> dict with keys AR, p_feasible, p_optimal
              e.g. {'PC-QAOA': {...}, 'PenaltyQAOA': {...}}
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
    ax.legend(fontsize=10, framealpha=0.4)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_outcome_distributions(counts: dict, constraints: list, n_x: int,
                               optimal_x: list = None, top_n: int = 20,
                               structural_constraints: list = None,
                               penalty_constraints: list = None,
                               title: str = 'Measurement distributions',
                               save_path: str = None) -> plt.Figure:
    """Side-by-side bar charts of top-N measurement outcomes, coloured by status.

    Parameters
    ----------
    counts : dict mapping method name -> {bitstring: shot_count}
    constraints : list of all constraint strings (decision-variable indexed)
    n_x : number of decision variables
    optimal_x : list of optimal decision bitstrings for colour coding
    top_n : number of top outcomes to show per method
    structural_constraints : list of structural constraint strings; when provided
        together with penalty_constraints, enables 5-category colour coding
    penalty_constraints : list of penalized constraint strings
    title : overall figure title
    save_path : if given, save figure to this path

    Colour scheme (5-category, when structural/penalty constraints are given)
    -------------------------------------------------------------------------
    foam  : Optimal (feasible + achieves optimal value)
    pine  : Fully feasible (all constraints satisfied)
    gold  : Structural ✓, Penalty ✗
    rose  : Structural ✗, Penalty ✓
    love  : All infeasible

    Falls back to 3-category (foam/pine/love) when constraints are not split.
    """
    pu.setup_style()
    five_category = (structural_constraints is not None
                     and penalty_constraints is not None)

    def _bar_color(bs):
        if optimal_x and bs in optimal_x:
            return pu._ROSE_PINE['foam']
        if five_category:
            feas_s = feasibility_check(bs, structural_constraints, n_x)
            feas_p = feasibility_check(bs, penalty_constraints, n_x)
            if feas_s and feas_p:
                return pu._ROSE_PINE['pine']
            if feas_s:
                return pu._ROSE_PINE['gold']
            if feas_p:
                return pu._ROSE_PINE['rose']
            return pu._ROSE_PINE['love']
        if feasibility_check(bs, constraints, n_x):
            return pu._ROSE_PINE['pine']
        return pu._ROSE_PINE['love']

    method_names = list(counts.keys())
    fig, axes = plt.subplots(1, len(method_names),
                             figsize=(7 * len(method_names), 5), sharey=False)
    if len(method_names) == 1:
        axes = [axes]

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

    if five_category:
        legend_patches = [
            mpatches.Patch(color=pu._ROSE_PINE['foam'], label='Optimal'),
            mpatches.Patch(color=pu._ROSE_PINE['pine'], label='All feasible'),
            mpatches.Patch(color=pu._ROSE_PINE['gold'], label='Structural ✓  Penalty ✗'),
            mpatches.Patch(color=pu._ROSE_PINE['rose'], label='Structural ✗  Penalty ✓'),
            mpatches.Patch(color=pu._ROSE_PINE['love'], label='All infeasible'),
        ]
        ncol = 5
    else:
        legend_patches = [
            mpatches.Patch(color=pu._ROSE_PINE['foam'], label='Optimal'),
            mpatches.Patch(color=pu._ROSE_PINE['pine'], label='Feasible'),
            mpatches.Patch(color=pu._ROSE_PINE['love'], label='Infeasible'),
        ]
        ncol = 3

    fig.legend(handles=legend_patches, loc='upper center', ncol=ncol,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.05)

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_layer_sweep(df: pd.DataFrame, title: str = 'Layer sweep',
                     save_path: str = None) -> plt.Figure:
    """Three-panel plot: AR, P(feasible), P(optimal) vs QAOA layers.

    Parameters
    ----------
    df : DataFrame with columns method, layer, AR, p_feasible, p_optimal
    """
    pu.setup_style()
    metrics = [('AR', 'AR'), ('p_feasible', '$P(\\mathrm{feas})$'), ('p_optimal', '$P(\\mathrm{opt})$')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (col, label) in zip(axes, metrics):
        for method, grp in df.groupby('method'):
            color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
            grp = grp.sort_values('layer')
            ax.plot(grp['layer'], grp[col], marker='o', label=method, color=color)
        ax.set_xlabel('QAOA layers ($p$)')
        ax.set_ylabel(label)
        ax.set_xticks(sorted(df['layer'].unique()))

    axes[0].legend(fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1x2 summary figures (replace 2x2 all/feas/matched/matched_feas variants)
# ─────────────────────────────────────────────────────────────────────────────

_OV_LS = {'disjoint': '-', 'overlapping': '--'}   # linestyle by overlap type

_LAYER_STYLES = {
    1: dict(marker='^', alpha=0.85, markersize=7),
    5: dict(marker='o', alpha=1.0, markersize=6),
}


def _apply_layer_style(ax_plot_kwargs, style, color, layer):
    """Merge layer style; open markers for p=1."""
    kw = dict(style)
    if layer == 1:
        kw['markerfacecolor'] = 'none'
        kw['markeredgecolor'] = color
    return kw


def _add_grouped_legend(ax, methods=None, overlaps=None, layers=None,
                        loc='upper right', color_map=None):
    """Structured legend: color = method, linestyle = structure, marker = depth."""
    from matplotlib.lines import Line2D
    handles = []
    muted = pu._ROSE_PINE.get('muted', '#6e6a86')

    if methods:
        for m in methods:
            color = ((color_map or {}).get(m)
                     or pu.METHOD_COLORS.get(m, pu._ROSE_PINE['subtle']))
            handles.append(Line2D([0], [0], color=color, linewidth=2, label=m))

    if overlaps:
        for ov in overlaps:
            ls = _OV_LS.get(ov, '-')
            handles.append(Line2D([0], [0], color=muted, linestyle=ls,
                                  linewidth=2, label=ov))

    if layers:
        for layer in layers:
            style = _LAYER_STYLES.get(layer, _LAYER_STYLES[5])
            h = Line2D([0], [0], color=muted, linestyle='-',
                       marker=style['marker'], markersize=style['markersize'],
                       label=f'$p={layer}$')
            if layer == 1:
                h.set_markerfacecolor('none')
                h.set_markeredgecolor(muted)
            handles.append(h)

    ax.legend(handles=handles, fontsize=10, loc=loc,
              handlelength=2.5, framealpha=0.4)


def add_vcg_flag(df: pd.DataFrame, exact_lookup: dict) -> pd.DataFrame:
    """Add boolean 'has_vcg' column: True if any structural constraint uses a
    QAOA-trained (non-exact) VCG gadget according to *exact_lookup*.

    Safe to call on both all-layers and final-layer DataFrames; the column is
    added in-place on a copy so the original is not mutated.
    """
    from core.constraint_handler import normalize_constraint
    df = df.copy()

    def _has_vcg(row):
        for c in (row.get('structural_constraints') or []):
            key = normalize_constraint(c)
            if key in exact_lookup and not exact_lookup[key]:
                return True
        return False

    df['has_vcg'] = df.apply(_has_vcg, axis=1)
    return df


def _add_prep_type(df: pd.DataFrame, exact_lookup: dict = None) -> pd.DataFrame:
    """Add 'prep_type' col: 'VCG' if the problem uses a QAOA-trained VCG gadget.

    If *exact_lookup* is provided (mapping constraint_key -> is_exact bool, built
    from vcg_db.pkl), classification is based on whether any structural constraint
    maps to a non-exact DB entry.  Falls back to the constraint_type string heuristic
    ('knapsack' in type name) when the lookup is absent or the column is missing.
    """
    from core.constraint_handler import normalize_constraint
    df = df.copy()

    if 'has_vcg' in df.columns:
        df['prep_type'] = df['has_vcg'].map({True: 'VCG', False: 'exact'})
    elif exact_lookup is not None and 'structural_constraints' in df.columns:
        def _classify(row):
            for c in (row['structural_constraints'] or []):
                key = normalize_constraint(c)
                if key in exact_lookup and not exact_lookup[key]:
                    return 'VCG'
            return 'exact'
        df['prep_type'] = df.apply(_classify, axis=1)
    else:
        df['prep_type'] = df['constraint_type'].apply(
            lambda ct: 'VCG' if 'knapsack' in str(ct).lower() else 'exact'
        )
    return df


def _panel(ax, df, metric, ylabel, group_col='method', threshold=None):
    """Plot mean ± std of metric vs n_x; group_col is 'method' or 'overlap_type'."""
    nx_vals = sorted(df['n_x'].unique())
    for gval, grp in df.groupby(group_col):
        if group_col == 'method':
            color = pu.METHOD_COLORS.get(gval, pu._ROSE_PINE['subtle'])
            label = gval
        else:
            color = pu._ROSE_PINE['pine'] if gval == 'disjoint' else pu._ROSE_PINE['iris']
            label = gval.capitalize()
        means = grp.groupby('n_x')[metric].mean()
        stds = grp.groupby('n_x')[metric].std().fillna(0)
        ax.plot(nx_vals, [means.get(n, np.nan) for n in nx_vals],
                marker='o', label=label, color=color)
        ax.fill_between(
            nx_vals,
            [max(0.0, means.get(n, 0) - stds.get(n, 0)) for n in nx_vals],
            [min(1.0, means.get(n, 0) + stds.get(n, 0)) for n in nx_vals],
            color=color, alpha=0.15,
        )
    if threshold is not None:
        ax.axhline(threshold, color=pu._ROSE_PINE['muted'],
                   linestyle='--', linewidth=1, label=f'threshold ({threshold})')
    ax.set_xlabel('$n$')
    ax.set_ylabel(ylabel)
    ax.set_xticks(nx_vals)
    ax.legend(fontsize=10, framealpha=0.4)


def _split_panel(ax, df, metric, ylabel, threshold=None, clip=(0.0, 1.0)):
    """4-line panel: both methods × {disjoint, overlapping}.

    Color = method (amethyst/blue), linestyle = overlap type (solid/dashed).
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
    if threshold is not None:
        ax.axhline(threshold, color=pu._ROSE_PINE['muted'],
                   linestyle=':', linewidth=1, label=f'threshold ({threshold})')
    ax.set_xlabel('$n$')
    ax.set_ylabel(ylabel)
    ax.set_xticks(nx_vals)
    leg = ax.legend(fontsize=10, ncol=2, handlelength=2.5, framealpha=0.4)
    for h in leg.legend_handles:
        h.set_marker('')


def plot_p_feasible_with_overlap_split(
    df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """1x2: (a) P(feas) vs n_x both methods; (b) both methods by overlap type (4 lines)."""
    pu.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    _panel(axes[0], df, 'p_feasible', '$P(\\mathrm{feas})$')
    axes[0].set_title('$P(\\mathrm{feas})$ vs $n$')

    _split_panel(axes[1], df, 'p_feasible', '$P(\\mathrm{feas})$')
    axes[1].set_title('$P(\\mathrm{feas})$: disjoint (—) vs overlapping (- -)')

    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_p_optimal_with_overlap_split(
    df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """1x2: (a) P(opt) vs n_x both methods; (b) P(opt) PC-QAOA by overlap type."""
    pu.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    _panel(axes[0], df, 'p_optimal', '$P(\\mathrm{opt})$')
    axes[0].set_title('$P(\\mathrm{opt})$ vs $n$')

    _split_panel(axes[1], df, 'p_optimal', '$P(\\mathrm{opt})$', clip=(0.0, 1.0))
    axes[1].set_title('$P(\\mathrm{opt})$: disjoint (—) vs overlapping (- -)')

    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_layers_to_threshold_with_overlap_split(
    df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """1x2: (a) layers-to-threshold bar chart; (b) per-overlap_type version.

    df must have n_layers = first layer where P(feas)>=0.75 per experiment
    (i.e. the convergence DataFrame from _build_convergence_df).
    """
    from analyze_results.plot_ar import plot_layers_to_threshold as _ltt
    pu.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    def _bar_panel(ax, subdf, title):
        methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in subdf['method'].values]
        colors = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
        p_max = int(subdf['n_layers'].max())
        x_conv = np.arange(1, p_max + 1)
        x_dnm = p_max + 1
        bar_w = 0.35

        for i, (method, color) in enumerate(zip(methods, colors)):
            sub = subdf[subdf['method'] == method]
            offset = (i - (len(methods) - 1) / 2) * bar_w
            counts = (sub[sub['p_feasible'] >= 0.75]['n_layers']
                      .value_counts()
                      .reindex(range(1, p_max + 1), fill_value=0)
                      if 'p_feasible' in sub.columns
                      else sub['n_layers'].value_counts().reindex(range(1, p_max + 1), fill_value=0))
            bars = ax.bar(x_conv + offset, counts.values, width=bar_w,
                          color=color, edgecolor='white', linewidth=0.7,
                          alpha=0.85, zorder=3, label=method)
            for bar, cnt in zip(bars, counts.values):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3, str(cnt),
                            ha='center', va='bottom', fontsize=7, zorder=4)
            dnm = ((sub['n_layers'] == p_max)
                   & (~(sub['p_feasible'] >= 0.75) if 'p_feasible' in sub.columns
                      else pd.Series([True] * len(sub), index=sub.index))).sum()
            dnm_bar = ax.bar(x_dnm + offset, dnm, width=bar_w,
                             color=color, edgecolor='white', linewidth=0.7,
                             alpha=0.45, hatch='//', zorder=3)
            if dnm > 0:
                ax.text(dnm_bar[0].get_x() + dnm_bar[0].get_width() / 2,
                        dnm + 0.3, str(dnm),
                        ha='center', va='bottom', fontsize=7, zorder=4)

        ax.set_xticks(list(x_conv) + [x_dnm])
        ax.set_xticklabels([str(p) for p in range(1, p_max + 1)] + ['Did not\nmeet'])
        ax.axvline(x=p_max + 0.5, color=pu._ROSE_PINE['muted'],
                   linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xlabel('QAOA layers $p$ at convergence')
        ax.set_ylabel('Experiment count')
        ax.set_title(title)
        ax.legend(fontsize=10, framealpha=0.4)

    _bar_panel(axes[0], df, 'Layers to $P(\\mathrm{feas}) \\geq 0.75$')

    # Panel (b): both methods × {disjoint, overlapping} — 4 bar groups
    # Use method color + hatch for overlap_type
    if 'overlap_type' in df.columns:
        p_max = int(df['n_layers'].max())
        x_conv = np.arange(1, p_max + 1)
        x_dnm = p_max + 1
        methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df['method'].values]
        ov_types = ['disjoint', 'overlapping']
        hatches = {'disjoint': '', 'overlapping': '//'}
        n_groups = len(methods) * len(ov_types)
        bar_w = 0.8 / n_groups
        idx = 0
        for method in methods:
            color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['muted'])
            for ov in ov_types:
                grp = df[(df['method'] == method) & (df['overlap_type'] == ov)]
                if grp.empty:
                    idx += 1
                    continue
                offset = (idx - (n_groups - 1) / 2) * bar_w
                counts = (grp[grp['p_feasible'] >= 0.75]['n_layers']
                          .value_counts()
                          .reindex(range(1, p_max + 1), fill_value=0)
                          if 'p_feasible' in grp.columns
                          else grp['n_layers'].value_counts().reindex(range(1, p_max + 1), fill_value=0))
                dnm = int(((grp['n_layers'] == p_max)
                           & ~(grp['p_feasible'] >= 0.75 if 'p_feasible' in grp.columns
                               else pd.Series(False, index=grp.index))).sum())
                label = f'{method} ({ov})'
                bars = axes[1].bar(x_conv + offset, counts.values, width=bar_w,
                                   color=color, hatch=hatches[ov],
                                   edgecolor='white', linewidth=0.5,
                                   alpha=0.85, zorder=3, label=label)
                if dnm > 0:
                    dnm_bar = axes[1].bar(x_dnm + offset, dnm, width=bar_w,
                                          color=color, hatch=hatches[ov],
                                          edgecolor='white', linewidth=0.5,
                                          alpha=0.45, zorder=3)
                idx += 1
        axes[1].set_xticks(list(x_conv) + [x_dnm])
        axes[1].set_xticklabels([str(p) for p in range(1, p_max + 1)] + ['Did not\nmeet'])
        axes[1].axvline(x=p_max + 0.5, color=pu._ROSE_PINE['muted'],
                        linewidth=0.8, linestyle='--', alpha=0.6)
        axes[1].set_xlabel('QAOA layers $p$ at convergence')
        axes[1].set_ylabel('Experiment count')
        axes[1].set_title('Convergence: disjoint (solid) vs overlapping (hatched)')
        axes[1].legend(fontsize=10, ncol=2)

    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def _plot_method_panel(ax, df_all, metric, threshold=None, layers=(1, 5)):
    """method-split panel: one line per (method, layer), p=1 open/p=5 filled."""
    nx_vals = sorted(df_all['n_x'].unique())
    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df_all['method'].values]
    for method in methods:
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        for layer in layers:
            sub = df_all[(df_all['method'] == method) & (df_all['layer'] == layer)]
            if sub.empty:
                continue
            style = _apply_layer_style({}, _LAYER_STYLES.get(layer, _LAYER_STYLES[5]), color, layer)
            means = sub.groupby('n_x')[metric].mean()
            stds = sub.groupby('n_x')[metric].std().fillna(0)
            ax.plot(nx_vals, [means.get(n, np.nan) for n in nx_vals],
                    color=color, label=f'{method} ($p={layer}$)', **style)
            ax.fill_between(
                nx_vals,
                [np.clip(means.get(n, 0) - stds.get(n, 0), 0, 1) for n in nx_vals],
                [np.clip(means.get(n, 0) + stds.get(n, 0), 0, 1) for n in nx_vals],
                color=color, alpha=0.10,
            )
    if threshold is not None:
        ax.axhline(threshold, color=pu._ROSE_PINE['muted'],
                   linestyle=':', linewidth=1, label=f'threshold ({threshold})')


def _plot_overlap_panel(ax, df_all, metric, threshold=None, layers=(1, 5)):
    """overlap-split panel: (method × overlap_type × layer), p=1+p=5."""
    nx_vals = sorted(df_all['n_x'].unique())
    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df_all['method'].values]
    for method in methods:
        color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['subtle'])
        mgrp = df_all[df_all['method'] == method]
        for ov, ogrp in mgrp.groupby('overlap_type'):
            ls = _OV_LS.get(ov, '-')
            for layer in layers:
                sub = ogrp[ogrp['layer'] == layer]
                if sub.empty:
                    continue
                style = _apply_layer_style({}, _LAYER_STYLES.get(layer, _LAYER_STYLES[5]), color, layer)
                style['linestyle'] = ls
                means = sub.groupby('n_x')[metric].mean()
                stds = sub.groupby('n_x')[metric].std().fillna(0)
                ax.plot(nx_vals, [means.get(n, np.nan) for n in nx_vals],
                        color=color, label=f'{method} ({ov}, $p={layer}$)', **style)
                ax.fill_between(
                    nx_vals,
                    [np.clip(means.get(n, 0) - stds.get(n, 0), 0, 1) for n in nx_vals],
                    [np.clip(means.get(n, 0) + stds.get(n, 0), 0, 1) for n in nx_vals],
                    color=color, alpha=0.07,
                )
    if threshold is not None:
        ax.axhline(threshold, color=pu._ROSE_PINE['muted'],
                   linestyle=':', linewidth=1, label=f'threshold ({threshold})')


def _plot_prep_panel(ax, df_all, metric, threshold=None, layers=(1, 5),
                     exact_lookup=None):
    """prep-split panel: PC-QAOA only, VCG vs exact × disjoint/overlapping, p=1+p=5.

    Color = prep type (VCG / exact).
    Linestyle = overlap type (solid = disjoint, dashed = overlapping) when
    the 'overlap_type' column is present; otherwise always solid.
    """
    df = _add_prep_type(df_all[df_all['method'] == 'PC-QAOA'], exact_lookup=exact_lookup)
    nx_vals = sorted(df['n_x'].unique())
    prep_colors = {
        'VCG': pu._ROSE_PINE.get('iris', '#907aa9'),
        'exact': '#e8976f',
    }
    has_splits = 'overlap_type' in df.columns and df['overlap_type'].nunique() > 1
    overlap_vals = sorted(df['overlap_type'].unique()) if has_splits else [None]

    for prep in ['VCG', 'exact']:
        color = prep_colors[prep]
        prep_df = df[df['prep_type'] == prep]
        for ov in overlap_vals:
            ls = _OV_LS.get(ov, '-') if ov is not None else '-'
            ov_df = prep_df[prep_df['overlap_type'] == ov] if ov is not None else prep_df
            for layer in layers:
                sub = ov_df[ov_df['layer'] == layer]
                if sub.empty:
                    continue
                style = _apply_layer_style({}, _LAYER_STYLES.get(layer, _LAYER_STYLES[5]), color, layer)
                style['linestyle'] = ls
                means = sub.groupby('n_x')[metric].mean()
                stds = sub.groupby('n_x')[metric].std().fillna(0)
                lbl = f'{prep}, {ov}, $p={layer}$' if ov is not None else f'{prep}, $p={layer}$'
                ax.plot(nx_vals, [means.get(n, np.nan) for n in nx_vals],
                        color=color, label=lbl, **style)
                ax.fill_between(
                    nx_vals,
                    [np.clip(means.get(n, 0) - stds.get(n, 0), 0, 1) for n in nx_vals],
                    [np.clip(means.get(n, 0) + stds.get(n, 0), 0, 1) for n in nx_vals],
                    color=color, alpha=0.08,
                )
    if threshold is not None:
        ax.axhline(threshold, color=pu._ROSE_PINE['muted'],
                   linestyle=':', linewidth=1, label=f'threshold ({threshold})')


def _simplify_constraint_type(ct: str) -> str:
    """Reduce a '+'-joined combination to the dominant high-level family."""
    families = set(ct.split('+'))
    if 'quadratic_knapsack' in families:
        return 'Quadratic knapsack'
    if 'knapsack' in families:
        return 'Knapsack'
    if 'flow' in families:
        return 'Flow'
    if 'independent_set' in families:
        return 'Independent set'
    if 'assignment' in families:
        return 'Assignment'
    return 'Cardinality'


def _classify_family(row) -> str:
    """Row-level family classifier; splits cardinality into = vs ≤/≥."""
    ct = str(row.get('constraint_type', ''))
    families = set(ct.split('+'))
    if 'quadratic_knapsack' in families:
        return 'Quadratic knapsack'
    if 'knapsack' in families:
        return 'Knapsack'
    if 'flow' in families:
        return 'Flow'
    if 'independent_set' in families:
        return 'Independent set'
    if 'assignment' in families:
        return 'Assignment'
    # Cardinality: inspect actual constraint strings to distinguish = from ≤/≥
    constraints = []
    for col in ('structural_constraints', 'penalty_constraints'):
        val = row.get(col, [])
        if isinstance(val, list):
            constraints.extend(val)
        elif isinstance(val, str):
            try:
                import ast
                constraints.extend(ast.literal_eval(val))
            except Exception:
                pass
    has_ineq = any(('<=' in c or '>=' in c) for c in constraints)
    return 'Cardinality (≤/≥)' if has_ineq else 'Cardinality (=)'


def plot_p_feasible_by_family(
    df: pd.DataFrame,
    df_all: pd.DataFrame = None,
    save_path: str = None,
) -> plt.Figure:
    """1x2 grouped bar chart: P(feas) by constraint family, one bar group per n_x.

    Bars show p=5 (final layer).  If df_all is provided, p=1 mean is overlaid as
    a horizontal tick marker on each bar.  Error bars capped at [0, 1].
    """
    pu.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    df = df.copy()
    df['family'] = df.apply(_classify_family, axis=1)
    family_order = ['Cardinality (=)', 'Cardinality (≤/≥)', 'Knapsack',
                    'Quadratic knapsack', 'Flow', 'Assignment', 'Independent set']
    # Paul Tol "bright" — colorblind-friendly, 7 perceptually distinct colors
    family_colors = {
        'Cardinality (=)': '#4477AA',
        'Cardinality (≤/≥)': '#66CCEE',
        'Knapsack': '#228833',
        'Quadratic knapsack': '#CCBB44',
        'Flow': '#EE6677',
        'Assignment': '#AA3377',
        'Independent set': '#BBBBBB',
    }
    nx_vals = sorted(df['n_x'].unique())
    n_fam = len(family_order)
    bar_w = 0.8 / n_fam
    x = np.arange(len(nx_vals))

    # Use all-layers data so bars (p=5) are always >= ticks (p=1)
    if df_all is not None and not df_all.empty and 'layer' in df_all.columns:
        p_max = int(df_all['layer'].max())
        df_p5 = df_all[df_all['layer'] == p_max].copy()
        df_p1 = df_all[df_all['layer'] == 1].copy()
        df_p5['family'] = df_p5.apply(_classify_family, axis=1)
        df_p1['family'] = df_p1.apply(_classify_family, axis=1)
    else:
        df_p5 = df.copy()
        df_p1 = None

    for ax, method in zip(axes, ['PC-QAOA', 'PenaltyQAOA']):
        sub5 = df_p5[df_p5['method'] == method]
        sub1 = df_p1[df_p1['method'] == method] if df_p1 is not None else None

        for i, fam in enumerate(family_order):
            grp5 = sub5[sub5['family'] == fam]
            color = family_colors[fam]
            offset = (i - (n_fam - 1) / 2) * bar_w
            centers = x + offset

            means = np.array([grp5[grp5['n_x'] == nx]['p_feasible'].mean()
                              for nx in nx_vals])
            stds = np.array([grp5[grp5['n_x'] == nx]['p_feasible'].std()
                             for nx in nx_vals])
            yerr_lo = np.minimum(stds, means)
            yerr_hi = np.minimum(stds, 1 - means)

            ax.bar(centers, means, bar_w * 0.9,
                   yerr=[yerr_lo, yerr_hi],
                   color=color, label=fam, alpha=0.85,
                   error_kw=dict(capsize=2, elinewidth=0.8,
                                 ecolor=pu._ROSE_PINE['subtle']))

            # p=1 tick markers
            if sub1 is not None:
                grp1 = sub1[sub1['family'] == fam]
                p1_means = np.array([grp1[grp1['n_x'] == nx]['p_feasible'].mean()
                                     for nx in nx_vals])
                half = bar_w * 0.4
                for cx, py in zip(centers, p1_means):
                    if not np.isnan(py):
                        ax.hlines(py, cx - half, cx + half,
                                  colors=pu._ROSE_PINE['text'],
                                  linewidth=3.0, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f'${n}$' for n in nx_vals], fontsize=10)
        ax.set_xlabel('Problem size ($n$)', fontsize=10)
        ax.set_ylabel('$P(\\mathrm{feas})$', fontsize=10)
        ax.set_title(f'{method}: $P(\\mathrm{{feas}})$ by constraint family', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(0, 1.1)

    # Single shared legend to the right of the second panel
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, framealpha=0.4,
               loc='center left', bbox_to_anchor=(0.98, 0.5), ncol=1)
    fig.tight_layout(rect=[0, 0, 0.87, 1])
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metrics vs QAOA layers
# ─────────────────────────────────────────────────────────────────────────────

_NX_MARKERS = {4: 'o', 5: 's', 6: '^', 7: 'D', 8: 'v', 9: 'P', 10: '*'}
_NX_PALETTE = [
    '#1f6feb', '#b08800', '#e85c8a', '#8250df', '#1a7f37', '#cf222e', '#6e7681'
]


def plot_metric_vs_nx(
    df_all: pd.DataFrame,
    metric: str,
    ylabel: str,
    save_path: str = None,
    threshold: float = None,
    layers: tuple = (1, 5),
    title: str = None,
    legend_loc: str = 'upper right',
) -> plt.Figure:
    """Standalone: metric vs n_x, PC-QAOA vs PenaltyQAOA, p=1 and p=5."""
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df_all['method'].values]
    _plot_method_panel(ax, df_all, metric, threshold=threshold, layers=layers)
    ax.set_xlabel('Problem size ($n$)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df_all['n_x'].unique()))
    if title:
        ax.set_title(title)
    _add_grouped_legend(ax, methods=methods, layers=list(layers), loc=legend_loc)
    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_metric_vs_nx_split(
    df_all: pd.DataFrame,
    metric: str,
    ylabel: str,
    save_path: str = None,
    threshold: float = None,
    layers: tuple = (1, 5),
    title: str = None,
    legend_loc: str = 'upper right',
) -> plt.Figure:
    """Standalone: metric vs n_x, split by disjoint/overlapping, p=1 and p=5."""
    if 'overlap_type' not in df_all.columns:
        return None
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df_all['method'].values]
    overlaps = sorted(df_all['overlap_type'].unique())
    _plot_overlap_panel(ax, df_all, metric, threshold=threshold, layers=layers)
    ax.set_xlabel('Problem size ($n$)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df_all['n_x'].unique()))
    if title:
        ax.set_title(title)
    _add_grouped_legend(ax, methods=methods, overlaps=overlaps,
                        layers=list(layers), loc=legend_loc)
    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_metric_vcg_vs_exact(
    df_all: pd.DataFrame,
    metric: str,
    ylabel: str,
    save_path: str = None,
    threshold: float = None,
    layers: tuple = (1, 5),
    title: str = None,
    legend_loc: str = 'upper right',
    exact_lookup: dict = None,
) -> plt.Figure:
    """Standalone: metric vs n_x, PC-QAOA split by VCG/exact prep type, p=1 and p=5.

    When 'overlap_type' is present, linestyle encodes disjoint (solid) vs
    overlapping (dashed), matching the style of the other split plots.
    """
    if 'constraint_type' not in df_all.columns:
        return None
    pu.setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    prep_colors = {
        'VCG': pu._ROSE_PINE.get('iris', '#907aa9'),
        'exact': '#e8976f',
    }
    _plot_prep_panel(ax, df_all, metric, threshold=threshold, layers=layers,
                     exact_lookup=exact_lookup)
    ax.set_xlabel('Problem size ($n$)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df_all['n_x'].unique()))
    if title:
        ax.set_title(title)
    has_splits = 'overlap_type' in df_all.columns and df_all['overlap_type'].nunique() > 1
    overlaps = sorted(df_all['overlap_type'].unique()) if has_splits else None
    _add_grouped_legend(ax, methods=['VCG', 'exact'], overlaps=overlaps,
                        layers=list(layers), loc=legend_loc,
                        color_map=prep_colors)
    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_vcg_exact_counts(
    df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """Grouped bar: PC-QAOA problems per n by VCG/exact × disjoint/overlapping.

    Uses hatching to distinguish disjoint (solid) from overlapping (hatched)
    while keeping colour consistent for VCG vs exact.
    """
    if 'has_vcg' not in df.columns:
        return None

    has_splits = 'overlap_type' in df.columns and df['overlap_type'].nunique() > 1

    pu.setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    pc = df[df['method'] == 'PC-QAOA'].drop_duplicates('constraints_hash')
    nx_vals = sorted(pc['n_x'].unique())
    x = np.arange(len(nx_vals))

    prep_colors = {
        True:  pu._ROSE_PINE.get('iris', '#907aa9'),
        False: '#e8976f',
    }
    prep_labels = {True: 'VCG (QAOA)', False: 'Exact'}

    if has_splits:
        # 4 bars per n: VCG-disjoint, exact-disjoint, VCG-overlapping, exact-overlapping
        bar_width = 0.20
        combos = [
            (True,  'disjoint',    '',     'VCG — disjoint'),
            (False, 'disjoint',    '',     'Exact — disjoint'),
            (True,  'overlapping', '///',  'VCG — overlapping'),
            (False, 'overlapping', '///',  'Exact — overlapping'),
        ]
        n_combos = len(combos)
        offsets = np.linspace(-(n_combos - 1) / 2, (n_combos - 1) / 2, n_combos) * bar_width
        for (has_vcg, ot, hatch, label), offset in zip(combos, offsets):
            sub = pc[pc['overlap_type'] == ot]
            counts = [sub[(sub['n_x'] == nx) & (sub['has_vcg'] == has_vcg)].shape[0]
                      for nx in nx_vals]
            color = prep_colors[has_vcg]
            bars = ax.bar(x + offset, counts, width=bar_width,
                          color=color, hatch=hatch,
                          edgecolor='white' if not hatch else pu._ROSE_PINE.get('subtle', '#6e6a86'),
                          linewidth=0.7, label=label, zorder=3)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            str(cnt), ha='center', va='bottom', fontsize=7, zorder=4)
    else:
        bar_width = 0.35
        for i, has_vcg in enumerate([True, False]):
            counts = [pc[(pc['n_x'] == nx) & (pc['has_vcg'] == has_vcg)].shape[0]
                      for nx in nx_vals]
            offset = (i - 0.5) * bar_width
            bars = ax.bar(x + offset, counts, width=bar_width,
                          color=prep_colors[has_vcg], edgecolor='white',
                          linewidth=0.7, label=prep_labels[has_vcg], zorder=3)
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            str(cnt), ha='center', va='bottom', fontsize=8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f'${nx}$' for nx in nx_vals])
    ax.set_xlabel('Problem size $n$')
    ax.set_ylabel('Number of problems')
    ax.set_title('PC-QAOA problems by state-prep type')
    ax.legend(framealpha=1, fontsize=8)

    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


