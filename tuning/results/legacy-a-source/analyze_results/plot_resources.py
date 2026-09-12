"""
plot_resources.py -- Circuit resource and timing plots.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    ax.set_xlabel('$n$')
    ax.set_ylabel('Estimated shots')
    ax.set_title('Estimated shots vs $n$')
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

    ax.set_xlabel('$n$')
    ax.set_ylabel('Circuit depth')
    ax.set_title('Circuit depth vs $n$')
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


def plot_vcg_total_time(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Strip + median: total per-row wall-clock time, per constraint family, coloured by n_x.

    Total time = hamiltonian_time + optimize_time + counts_time for each row.
    Style matches vcg_db train_time plot.
    """
    pu.setup_style()

    time_cols = [c for c in ['hamiltonian_time', 'optimize_time', 'counts_time']
                 if c in df.columns]
    if not time_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No timing data', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    def _to_float(v):
        if v is None:
            return 0.0
        if isinstance(v, list):
            v = v[0]
        return float(v) if v is not None else 0.0

    df = df.copy()
    for col in time_cols:
        df[col] = df[col].apply(_to_float)
    df['_total'] = df[time_cols].sum(axis=1)

    NX_COLORS   = {3: pu._ROSE_PINE['pine'], 4: pu._ROSE_PINE['gold'], 5: pu._ROSE_PINE['iris']}
    NX_MARKERS  = {3: 'o', 4: 's', 5: '^'}
    FAMILY_LABELS = {'knapsack': 'Knapsack', 'quadratic_knapsack': 'Quadratic knapsack'}

    families = sorted(df['constraint_type'].unique()) if 'constraint_type' in df.columns else ['all']
    x_pos = {f: i for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(99)

    nx_vals = sorted(df['n_x'].unique()) if 'n_x' in df.columns else []
    for nx in nx_vals:
        sub = df[df['n_x'] == nx]
        for fam in families:
            vals = sub[sub['constraint_type'] == fam]['_total'].dropna().values \
                   if 'constraint_type' in df.columns else sub['_total'].dropna().values
            if len(vals) == 0:
                continue
            xi = x_pos[fam]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals,
                       color=NX_COLORS.get(nx, pu._ROSE_PINE['muted']),
                       marker=NX_MARKERS.get(nx, 'o'),
                       s=40, alpha=0.75, zorder=3,
                       label=f'$|\\text{{supp}}(c_k)|={nx}$' if fam == families[0] else '')

    for fam in families:
        xi = x_pos[fam]
        sub = df[df['constraint_type'] == fam] if 'constraint_type' in df.columns else df
        med = sub['_total'].median()
        ax.hlines(med, xi - 0.3, xi + 0.3, colors=pu._ROSE_PINE['text'], linewidth=2, zorder=4)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in families])
    ax.set_ylabel('Total time (s)')
    ax.set_title('VCG per-run wall-clock time by constraint family')

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), title='$|\\text{supp}(c_k)|$', framealpha=1,
              loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(seen))

    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_comparison_total_time(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Box + strip: total per-layer time per method (PC-QAOA vs PenaltyQAOA).

    Total time per row = hamiltonian_time + optimize_time + counts_time.
    """
    pu.setup_style()

    if 'method' not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No method column', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    time_cols = [c for c in ['hamiltonian_time', 'optimize_time', 'counts_time']
                 if c in df.columns]
    if not time_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No timing data', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    def _to_float(v):
        if v is None:
            return 0.0
        if isinstance(v, list):
            v = v[0]
        return float(v) if v is not None else 0.0

    df = df.copy()
    for col in time_cols:
        df[col] = df[col].apply(_to_float)
    df['_total'] = df[time_cols].sum(axis=1)

    methods  = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in df['method'].values]
    colors   = [pu.METHOD_COLORS.get(m, pu._ROSE_PINE['muted']) for m in methods]
    data     = [df[df['method'] == m]['_total'].dropna().values for m in methods]
    positions = list(range(len(methods)))

    fig, ax = plt.subplots(figsize=(6, 4))
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
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=25, alpha=0.75, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Total time per layer (s)')
    ax.set_title('Per-layer wall-clock time\nPC-QAOA vs PenaltyQAOA')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_total_time_vs_nx(
    df: pd.DataFrame,
    comp_ar: pd.DataFrame = None,
    save_path: str = None,
) -> plt.Figure:
    """Line plot: mean total solve time (hours) vs n_x, one line per method.

    For each unique experiment (task), the total solve time is:
    - The max ``cumulative_time`` across its layers (if column is present), OR
    - The sum of ``hamiltonian_time + optimize_time + counts_time`` across layers.

    Experiments that did NOT converge (p_feasible < 0.75 AND n_layers == p_max,
    identified via ``comp_ar``) are assigned 7*24*3600 s = 604800 s as their
    total time (assumed to take >7 days).

    Parameters
    ----------
    df : pd.DataFrame
        comparison_resources DataFrame (one row per method/task/layer).
    comp_ar : pd.DataFrame, optional
        comparison_ar DataFrame used to identify non-converged tasks.
        Joined on ['method', 'constraint_type', 'n_x', 'layer', 'angle_strategy'].
    save_path : str, optional
        File path to save the figure.
    """
    pu.setup_style()

    DNF_SECONDS = 7 * 24 * 3600  # 604800 s

    if 'method' not in df.columns or 'n_x' not in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, 'No method/n_x columns', transform=ax.transAxes,
                ha='center', va='center')
        if save_path:
            pu.save_fig(fig, save_path)
        return fig

    def _to_float(v):
        if v is None:
            return np.nan
        if isinstance(v, list):
            v = v[0]
        return float(v) if v is not None else np.nan

    df = df.copy()

    # Group columns that identify a unique instance
    id_cols = [c for c in ['method', 'qubo_string', 'constraint_type', 'n_x', 'n_c',
                            'angle_strategy', 'constraints_hash']
               if c in df.columns]

    if 'cumulative_time' in df.columns:
        df['cumulative_time'] = df['cumulative_time'].apply(_to_float)
        task_times = (
            df.groupby(id_cols)['cumulative_time']
            .max()
            .reset_index()
            .rename(columns={'cumulative_time': 'total_solve_time'})
        )
    else:
        time_cols = [c for c in ['hamiltonian_time', 'optimize_time', 'counts_time']
                     if c in df.columns]
        for col in time_cols:
            df[col] = df[col].apply(_to_float)
        df['_layer_time'] = df[time_cols].sum(axis=1) if time_cols else 0.0
        task_times = (
            df.groupby(id_cols)['_layer_time']
            .sum()
            .reset_index()
            .rename(columns={'_layer_time': 'total_solve_time'})
        )

    # Identify non-converged instances via comp_ar and assign 7-day ceiling
    if comp_ar is not None and not comp_ar.empty:
        comp_ar_c = comp_ar.copy()
        p_max = int(comp_ar_c['n_layers'].max())
        join_keys = [c for c in id_cols if c in comp_ar_c.columns]
        if 'p_feasible' in comp_ar_c.columns and 'n_layers' in comp_ar_c.columns:
            task_conv = (
                comp_ar_c.groupby(join_keys)
                .apply(lambda g: pd.Series({
                    'converged': (g['p_feasible'] >= 0.75).any(),
                    'n_layers_max': g['n_layers'].max(),
                }), include_groups=False)
                .reset_index()
            )
            task_conv['not_converged'] = (
                (~task_conv['converged']) & (task_conv['n_layers_max'] == p_max)
            )
            task_times = task_times.merge(
                task_conv[join_keys + ['not_converged']], on=join_keys, how='left'
            )
            task_times['not_converged'] = task_times['not_converged'].fillna(False)
            # Actual measured times are used for all instances; no ceiling override

    # Filter to comparison n_x range only (exclude VCG-only sizes)
    task_times = task_times[task_times['n_x'] >= 4]

    # 1x2 figure: (a) both methods combined, (b) both methods × overlap_type
    has_split = 'overlap_type' in task_times.columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    methods = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in task_times['method'].values]
    _OV_LS  = {'disjoint': '-', 'overlapping': '--'}

    def _time_lines(ax, sub_df, linestyle='-', label_suffix=''):
        for method in methods:
            color = pu.METHOD_COLORS.get(method, pu._ROSE_PINE['muted'])
            sub = sub_df[sub_df['method'] == method]
            if sub.empty:
                continue
            stats = (sub.groupby('n_x')['total_solve_time']
                     .agg(['mean', 'std']).reset_index())
            stats['std'] = stats['std'].fillna(0.0)
            mean_h = stats['mean'].values / 3600.0
            lo_h   = np.clip(mean_h - stats['std'].values / 3600.0, 0, None)
            hi_h   = mean_h + stats['std'].values / 3600.0
            nx_v   = stats['n_x'].values
            lbl    = method + (f' ({label_suffix})' if label_suffix else '')
            ax.plot(nx_v, mean_h, marker='o', color=color,
                    linestyle=linestyle, label=lbl, zorder=3)
            ax.fill_between(nx_v, lo_h, hi_h, color=color, alpha=0.12, zorder=2)

    # Panel (a): combined
    _time_lines(axes[0], task_times)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel('$n$')
    axes[0].set_ylabel('Total solve time (hours)')
    axes[0].set_title('Total solve time vs $n$')
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[0].legend(framealpha=1, fontsize=8)

    # Panel (b): split by overlap_type (4 lines)
    if has_split:
        for ov in ['disjoint', 'overlapping']:
            sub_ov = task_times[task_times['overlap_type'] == ov]
            _time_lines(axes[1], sub_ov,
                        linestyle=_OV_LS[ov], label_suffix=ov)
    else:
        _time_lines(axes[1], task_times)
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel('$n$')
    axes[1].set_ylabel('Total solve time (hours)')
    axes[1].set_title('Total solve time: disjoint (—) vs overlapping (- -)')
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[1].legend(framealpha=1, fontsize=7, ncol=2)

    fig.tight_layout()
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
    ax.set_xlabel('$n$')
    ax.set_ylabel('Time (s)')
    ax.set_title('Mean time breakdown by $n$')
    ax.legend()

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def plot_total_time_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Box + strip: cumulative solve time per task, PC-QAOA vs PenaltyQAOA.

    ``df`` must contain a ``method`` column with values 'PC-QAOA' / 'PenaltyQAOA'
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

    methods = ['PC-QAOA', 'PenaltyQAOA']
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


def plot_circuit_resources_vs_nx(
    df: pd.DataFrame,
    p_layers: list[int] | None = None,
    save_path: str = None,
) -> plt.Figure:
    """Three-panel: (a) qubit count, (b) 2Q gates at fixed p, (c) 2Q gates vs p, vs n_x."""
    if p_layers is None:
        p_layers = [1, 5]

    pu.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    nx_vals = sorted(df['n_x'].unique())
    h_color = pu.METHOD_COLORS['PC-QAOA']
    p_color = pu.METHOD_COLORS['PenaltyQAOA']

    def _fill(ax, nx_vals, means, stds, color, alpha=0.15):
        ax.fill_between(
            nx_vals,
            [max(0, means[n] - stds[n]) for n in nx_vals],
            [means[n] + stds[n] for n in nx_vals],
            color=color, alpha=alpha,
        )

    # ── Panel (a): qubits ─────────────────────────────────────────────────
    ax = axes[0]
    for col, label, color in [
        ('n_qubits_pc', 'PC-QAOA', h_color),
        ('n_qubits_p', 'PenaltyQAOA', p_color),
    ]:
        means = df.groupby('n_x')[col].mean()
        stds  = df.groupby('n_x')[col].std().fillna(0)
        ax.plot(nx_vals, [means[n] for n in nx_vals], marker='o', label=label, color=color)
        _fill(ax, nx_vals, means, stds, color)
    ax.set_xlabel('Problem size ($n$)')
    ax.set_ylabel('Qubits')
    ax.set_title('Circuit width')
    ax.legend(fontsize=10, framealpha=0.4, loc='upper left')

    # ── Panel (b): 2Q gates at fixed p values vs n_x ─────────────────────
    ax = axes[1]
    linestyles = ['-', '--']
    for ls, p in zip(linestyles, p_layers):
        for sp_col, lay_col, color, method in [
            ('sp_2q_pc', 'layer_2q_pc', h_color, 'PC-QAOA'),
            ('sp_2q_p', 'layer_2q_p', p_color, 'PenaltyQAOA'),
        ]:
            vals = df[sp_col] + p * df[lay_col]
            tmp = pd.DataFrame({'n_x': df['n_x'], 'g': vals})
            means = tmp.groupby('n_x')['g'].mean()
            stds  = tmp.groupby('n_x')['g'].std().fillna(0)
            ax.plot(nx_vals, [means[n] for n in nx_vals],
                    marker='o', linestyle=ls, label=f'{method} $p={p}$', color=color)
            _fill(ax, nx_vals, means, stds, color, alpha=0.08)
    ax.set_xlabel('Problem size ($n$)')
    ax.set_ylabel('Two-qubit gates')
    ax.set_title('Two-qubit gate count vs $n$')
    ax.legend(fontsize=10, framealpha=0.4, loc='upper left')

    fig.tight_layout()
    if save_path:
        pu.save_fig(fig, save_path)
    return fig
