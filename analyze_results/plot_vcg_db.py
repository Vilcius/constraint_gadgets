"""
plot_vcg_db.py -- Generate VCG gadget-creation summary plots from vcg_db.pkl.

Plots produced
--------------
1. vcg_ar_by_type.png     — AR distribution by constraint family (all = 1.0)
2. vcg_entropy_by_type.png — S_norm (entropy) by constraint family, colour=n_x
3. vcg_layers_by_type.png  — Number of QAOA layers by constraint family
4. vcg_layers_vs_nx.png   — Layers distribution vs n_x (box + strip)
5. vcg_entropy_vs_layers.png — S_norm vs layers coloured by constraint family

Usage
-----
    python analyze_results/plot_vcg_db.py \
        --db gadgets/vcg_db.pkl \
        --output-dir analysis_output/figures/vcg_db/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from analyze_results.plot_utils import setup_style, save_fig, CONSTRAINT_COLORS, _ROSE_PINE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify(key: str) -> str:
    """Classify a constraint key as 'knapsack' or 'quadratic_knapsack'."""
    return 'quadratic_knapsack' if re.search(r'x_\d+\*x_\d+', key) else 'knapsack'


def load_db_as_df(db_path: str) -> pd.DataFrame:
    with open(db_path, 'rb') as f:
        db = pickle.load(f)

    rows = []
    for key, entry in db.items():
        rows.append({
            'constraint_key':              key,
            'constraint_type':             _classify(key),
            'constraints':                 entry.get('constraints'),
            'n_x':                         int(entry['n_x']),
            'n_feasible':                  int(entry['n_feasible']) if entry.get('n_feasible') is not None else np.nan,
            'ar':                          float(entry['ar']),
            'entropy':                     float(entry['entropy']) if entry['entropy'] is not None else np.nan,
            'converged':                   bool(entry['converged']) if 'converged' in entry else None,
            'n_layers':                    int(entry['n_layers']),
            'train_time':                  float(entry['train_time']) if entry.get('train_time') is not None else np.nan,
            'opt_angles':                  entry.get('opt_angles'),
            'single_feasible_bitstring':   entry.get('single_feasible_bitstring'),
            'dicke_superposition_weights': entry.get('dicke_superposition_weights'),
            'is_exact':                    entry.get('opt_angles') is None,
            'training_history':            entry.get('training_history'),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_LABELS = {
    'knapsack': 'Knapsack',
    'quadratic_knapsack': 'Quadratic knapsack',
}
NX_MARKERS = {3: 'o', 4: 's', 5: '^'}
NX_COLORS = {3: _ROSE_PINE['pine'], 4: _ROSE_PINE['gold'], 5: _ROSE_PINE['iris']}


def plot_ar_by_type(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: mean AR per family. (Should all be 1.0.)"""
    df = df[~df['is_exact']]
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    families = sorted(df['constraint_type'].unique())
    means = [df[df['constraint_type'] == f]['ar'].mean() for f in families]
    colors = [CONSTRAINT_COLORS.get(f, _ROSE_PINE['muted']) for f in families]
    labels = [FAMILY_LABELS.get(f, f) for f in families]

    bars = ax.bar(labels, means, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Approximation ratio (AR)')
    ax.set_title('VCG gadget AR by constraint family')
    ax.axhline(1.0, color=_ROSE_PINE['muted'], linewidth=0.8, linestyle='--', zorder=2)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    save_fig(fig, os.path.join(out_dir, 'vcg_ar_by_type.png'))
    print(f'  Saved vcg_ar_by_type.png')


def plot_entropy_by_type(df: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart: mean S_norm per (|S_k|, constraint family) with ±1 std error bars."""
    df = df[~df['is_exact']]
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    nx_vals = sorted(df['n_x'].unique())
    families = sorted(df['constraint_type'].unique())
    n_fam = len(families)
    bar_width = 0.35
    x = np.arange(len(nx_vals))

    for i, fam in enumerate(families):
        means, yerr_lo, yerr_hi = [], [], []
        for nx in nx_vals:
            vals = df[(df['constraint_type'] == fam) & (df['n_x'] == nx)]['entropy'].dropna()
            m = vals.mean()
            s = vals.std() if len(vals) > 1 else 0.0
            means.append(m)
            yerr_lo.append(min(s, m))          # clamp lower bar to 0
            yerr_hi.append(min(s, 1.0 - m))    # clamp upper bar to 1
        offset = (i - (n_fam - 1) / 2) * bar_width
        ax.bar(x + offset, means, width=bar_width,
               color=CONSTRAINT_COLORS.get(fam, _ROSE_PINE['muted']),
               edgecolor='white', linewidth=0.7,
               label=FAMILY_LABELS.get(fam, fam), zorder=3,
               yerr=[yerr_lo, yerr_hi], capsize=4,
               error_kw=dict(elinewidth=1, capthick=1, ecolor=_ROSE_PINE['text']))

    ax.set_xticks(x)
    ax.set_xticklabels([f'${nx}$' for nx in nx_vals])
    ax.set_xlabel('$|\\text{supp}(c_k)|$')
    ax.set_ylabel('$\\mathcal{S}_{\\mathrm{norm}}$')
    ax.set_title('VCG gadget $\\mathcal{S}_{\\mathrm{norm}}$ by constraint family')
    ax.set_ylim(0, 1.15)
    ax.legend(framealpha=1)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'vcg_entropy_by_type.png'))
    print(f'  Saved vcg_entropy_by_type.png')


def plot_layers_by_type(df: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart: layer count distribution per family."""
    df = df[~df['is_exact']]
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    families = sorted(df['constraint_type'].unique())
    max_layers = int(df['n_layers'].max())

    bar_width = 0.35
    x = np.arange(1, max_layers + 1)

    for i, fam in enumerate(families):
        sub = df[df['constraint_type'] == fam]
        counts = sub['n_layers'].value_counts().reindex(range(1, max_layers + 1), fill_value=0)
        offset = (i - (len(families) - 1) / 2) * bar_width
        ax.bar(x + offset, counts.values,
               width=bar_width,
               color=CONSTRAINT_COLORS.get(fam, _ROSE_PINE['muted']),
               edgecolor='white', linewidth=0.7,
               label=FAMILY_LABELS.get(fam, fam),
               zorder=3)

    ax.set_xlabel('QAOA layers $p$')
    ax.set_ylabel('Number of gadgets')
    ax.set_title('VCG gadget training depth by constraint family')
    ax.set_xticks(x)
    ax.legend()

    save_fig(fig, os.path.join(out_dir, 'vcg_layers_by_type.png'))
    print(f'  Saved vcg_layers_by_type.png')


FAMILY_MARKERS = {
    'knapsack': 'o',
    'quadratic_knapsack': '^',
}


def plot_layers_vs_nx(df: pd.DataFrame, out_dir: str) -> None:
    """Box + strip: QAOA layers vs n_x, colour=n_x, shape=constraint family."""
    df = df[~df['is_exact']]
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))

    nx_vals = sorted(df['n_x'].unique())
    positions = list(range(len(nx_vals)))
    families = sorted(df['constraint_type'].unique())

    data_by_nx = [df[df['n_x'] == nx]['n_layers'].values for nx in nx_vals]

    bp = ax.boxplot(data_by_nx, positions=positions,
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color=_ROSE_PINE['text'], linewidth=2),
                    whiskerprops=dict(color=_ROSE_PINE['muted']),
                    capprops=dict(color=_ROSE_PINE['muted']),
                    flierprops=dict(marker='', alpha=0),
                    zorder=2)

    for patch, nx in zip(bp['boxes'], nx_vals):
        patch.set_facecolor(NX_COLORS[nx])
        patch.set_alpha(0.4)

    rng = np.random.default_rng(7)
    for pos, nx in zip(positions, nx_vals):
        sub_nx = df[df['n_x'] == nx]
        for fam in families:
            vals = sub_nx[sub_nx['constraint_type'] == fam]['n_layers'].values
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(pos + jitter, vals,
                       color=NX_COLORS[nx],
                       marker=FAMILY_MARKERS.get(fam, 'o'),
                       s=40, alpha=0.85, zorder=3)

    # Legend: one entry per family (shape only, neutral colour)
    legend_handles = [
        plt.scatter([], [], color=_ROSE_PINE['subtle'],
                    marker=FAMILY_MARKERS.get(fam, 'o'), s=40,
                    label=FAMILY_LABELS.get(fam, fam))
        for fam in families
    ]
    ax.legend(handles=legend_handles, framealpha=1,
              loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=len(families))

    ax.set_xticks(positions)
    ax.set_xticklabels([f'${nx}$' for nx in nx_vals])
    ax.set_xlabel('$|\\text{supp}(c_k)|$')
    ax.set_ylabel('QAOA layers $p$')
    ax.set_title('VCG training depth vs problem size')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'vcg_layers_vs_nx.png'))
    print(f'  Saved vcg_layers_vs_nx.png')


def plot_entropy_vs_layers(df: pd.DataFrame, out_dir: str) -> None:
    """Scatter: S_norm vs n_layers, coloured by constraint family."""
    df = df[~df['is_exact']]
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    rng = np.random.default_rng(13)
    for fam in sorted(df['constraint_type'].unique()):
        sub = df[df['constraint_type'] == fam].dropna(subset=['entropy'])
        jitter = rng.uniform(-0.15, 0.15, size=len(sub))
        ax.scatter(sub['n_layers'] + jitter, sub['entropy'],
                   color=CONSTRAINT_COLORS.get(fam, _ROSE_PINE['muted']),
                   s=45, alpha=0.8, zorder=3,
                   label=FAMILY_LABELS.get(fam, fam))

    ax.set_xlabel('QAOA layers $p$')
    ax.set_ylabel('$\\mathcal{S}_{\\mathrm{norm}}$')
    ax.set_title('Gadget entropy vs training depth')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()

    save_fig(fig, os.path.join(out_dir, 'vcg_entropy_vs_layers.png'))
    print(f'  Saved vcg_entropy_vs_layers.png')


def plot_convergence_scatter(df: pd.DataFrame, out_dir: str) -> None:
    """Scatter: best-layer vs entropy, star=converged, circle=not, colour=family."""
    df = df[~df['is_exact']]
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    rng = np.random.default_rng(17)
    families = sorted(df['constraint_type'].unique())

    for fam in families:
        color = CONSTRAINT_COLORS.get(fam, _ROSE_PINE['muted'])
        sub = df[df['constraint_type'] == fam].dropna(subset=['entropy'])

        conv     = sub[sub['converged'] == True]
        not_conv = sub[sub['converged'] != True]

        if len(not_conv) > 0:
            jitter = rng.uniform(-0.25, 0.25, size=len(not_conv))
            ax.scatter(not_conv['n_layers'] + jitter, not_conv['entropy'],
                       color=color, marker='o', s=50, alpha=0.8, zorder=3)

        if len(conv) > 0:
            jitter = rng.uniform(-0.25, 0.25, size=len(conv))
            ax.scatter(conv['n_layers'] + jitter, conv['entropy'],
                       color=color, marker='*', s=180, alpha=1.0, zorder=4,
                       edgecolors=color, linewidths=0.5)

    # 4-entry legend: one per family (colour) + circle/star (convergence)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CONSTRAINT_COLORS.get(f, _ROSE_PINE['muted']),
               markersize=8, label=FAMILY_LABELS.get(f, f))
        for f in families
    ] + [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_ROSE_PINE['subtle'],
               markersize=8, label='Not converged'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=_ROSE_PINE['subtle'],
               markersize=12, label='Converged'),
    ]
    ax.legend(handles=handles, framealpha=1, fontsize=8,
              loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)

    ax.set_xlabel('Best-entropy layer $p$')
    ax.set_ylabel('$\\mathcal{S}_{\\mathrm{norm}}$')
    ax.set_title('VCG training outcome')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'vcg_convergence_scatter.png'))
    print(f'  Saved vcg_convergence_scatter.png')


def plot_train_time(df: pd.DataFrame, out_dir: str) -> None:
    """Strip + median marker: training time per family, coloured by n_x."""
    df = df[~df['is_exact']]
    if 'train_time' not in df.columns or df['train_time'].isna().all():
        print('  [skip] vcg_train_time.png — no train_time data in DB')
        return
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    families = sorted(df['constraint_type'].unique())
    x_pos = {f: i for i, f in enumerate(families)}

    rng = np.random.default_rng(99)
    for nx in sorted(df['n_x'].unique()):
        sub = df[df['n_x'] == nx]
        for fam in families:
            vals = sub[sub['constraint_type'] == fam]['train_time'].dropna().values
            if len(vals) == 0:
                continue
            xi = x_pos[fam]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals,
                       color=NX_COLORS[nx], marker=NX_MARKERS[nx],
                       s=40, alpha=0.75, zorder=3,
                       label=f'$|\\text{{supp}}(c_k)|={nx}$' if fam == families[0] else '')

    # Median markers
    for fam in families:
        xi = x_pos[fam]
        med = df[df['constraint_type'] == fam]['train_time'].median()
        ax.hlines(med, xi - 0.3, xi + 0.3, colors=_ROSE_PINE['text'],
                  linewidth=2, zorder=4)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in families])
    ax.set_ylabel('Training time (s)')
    ax.set_title('VCG gadget training time by constraint family')

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), title='$|\\text{supp}(c_k)|$', framealpha=1)

    save_fig(fig, os.path.join(out_dir, 'vcg_train_time.png'))
    print(f'  Saved vcg_train_time.png')


def plot_circuit_resources(res_df: pd.DataFrame, out_dir: str) -> None:
    """Two-panel: (a) total gates, (b) two-qubit gates vs n_x, by family."""
    res_df = res_df[res_df['n_layers'] > 0]
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    nx_vals = sorted(res_df['n_x'].unique())
    families = sorted(res_df['constraint_type'].unique())

    panels = [
        (axes[0], 'sp_total', 'Total gates', 'State-prep: total gates'),
        (axes[1], 'sp_2q', 'Two-qubit gates', 'State-prep: two-qubit gates'),
    ]
    for ax, col, ylabel, title in panels:
        for fam in families:
            color = CONSTRAINT_COLORS.get(fam, _ROSE_PINE['subtle'])
            label = FAMILY_LABELS.get(fam, fam)
            sub = res_df[res_df['constraint_type'] == fam]
            means = sub.groupby('n_x')[col].mean()
            stds = sub.groupby('n_x')[col].std().fillna(0)
            ax.plot(nx_vals, [means.get(n, 0) for n in nx_vals],
                    marker='o', label=label, color=color)
            ax.fill_between(
                nx_vals,
                [max(0, means.get(n, 0) - stds.get(n, 0)) for n in nx_vals],
                [means.get(n, 0) + stds.get(n, 0) for n in nx_vals],
                color=color, alpha=0.15,
            )
        ax.set_xlabel('$|\\text{supp}(c_k)|$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(nx_vals)
        ax.legend()

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'vcg_circuit_resources.png'))
    print('  Saved vcg_circuit_resources.png')


def plot_vcg_layers_to_entropy_threshold(
    df: pd.DataFrame, out_dir: str, threshold: float = 0.9999
) -> None:
    """Grouped bar: training depth at convergence, by constraint family.
    Exact-prep gadgets are excluded — they trivially converge by construction.

    Converged = final entropy >= threshold. Gadgets that did not reach the
    threshold are collected in a 'Did not meet' bar (hashed).
    """
    df = df[~df['is_exact']]
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    families = sorted(df['constraint_type'].unique())
    colors = [CONSTRAINT_COLORS.get(f, _ROSE_PINE['muted']) for f in families]
    max_layers = int(df['n_layers'].max())
    bar_width = 0.35

    x_converged = np.arange(1, max_layers + 1)
    x_dnm = max_layers + 1

    for i, (fam, color) in enumerate(zip(families, colors)):
        sub = df[df['constraint_type'] == fam].copy()
        sub['converged'] = sub['entropy'] >= threshold
        offset = (i - (len(families) - 1) / 2) * bar_width

        counts = (sub[sub['converged']]['n_layers']
                  .value_counts()
                  .reindex(range(1, max_layers + 1), fill_value=0))
        bars = ax.bar(x_converged + offset, counts.values,
                      width=bar_width, color=color, edgecolor='white',
                      linewidth=0.7, alpha=0.85, zorder=3,
                      label=FAMILY_LABELS.get(fam, fam))
        for bar, cnt in zip(bars, counts.values):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(cnt), ha='center', va='bottom', fontsize=7, zorder=4)

        dnm_count = int((~sub['converged']).sum())
        dnm_bar = ax.bar(x_dnm + offset, dnm_count,
                         width=bar_width, color=color, edgecolor='white',
                         linewidth=0.7, alpha=0.45, hatch='//', zorder=3)
        if dnm_count > 0:
            ax.text(dnm_bar[0].get_x() + dnm_bar[0].get_width() / 2,
                    dnm_count + 0.3, str(dnm_count),
                    ha='center', va='bottom', fontsize=7, zorder=4)

    ax.axvline(x=max_layers + 0.5, color=_ROSE_PINE['muted'],
               linewidth=0.8, linestyle='--', alpha=0.6)

    all_x = list(x_converged) + [x_dnm]
    all_labels = [str(p) for p in range(1, max_layers + 1)] + ['Did not\nmeet']
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)
    ax.set_xlabel('QAOA layers $p$ at convergence')
    ax.set_ylabel('Number of gadgets')
    ax.set_title(
        f'VCG layers to $\\mathcal{{S}}_{{\\mathrm{{norm}}}} \\geq {threshold}$')
    ax.legend(framealpha=1)

    save_fig(fig, os.path.join(out_dir, 'vcg_layers_to_entropy_threshold.png'))
    print(f'  Saved vcg_layers_to_entropy_threshold.png')


def print_summary(df: pd.DataFrame) -> None:
    """Print count/mean/min/max statistics for the VCG database.

    Prints two tables to stdout: one grouped by ``constraint_type`` and one
    grouped by ``n_x``, covering the ``ar``, ``entropy``, and ``n_layers``
    columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`load_db_as_df`.
    """
    print('\n=== VCG Gadget DB Summary ===')
    print(f'Total gadgets: {len(df)}')
    if 'converged' in df.columns:
        print(f'Converged: {df["converged"].sum()} / {len(df)}')
    print(f'\nBy constraint type:')
    cols = [c for c in ['n_feasible', 'ar', 'entropy', 'n_layers'] if c in df.columns]
    print(df.groupby('constraint_type')[cols].agg(
        ['count', 'mean', 'min', 'max']).to_string())
    print(f'\nBy n_x:')
    print(df.groupby('n_x')[cols].agg(
        ['count', 'mean', 'min', 'max']).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point.  Load the VCG database, print a summary, and generate
    all standard plots via :func:`plot_ar_by_type`, :func:`plot_entropy_by_type`,
    :func:`plot_layers_by_type`, :func:`plot_layers_vs_nx`,
    :func:`plot_entropy_vs_layers`, and :func:`plot_train_time`."""
    parser = argparse.ArgumentParser(description='Plot VCG gadget creation summary.')
    parser.add_argument('--db', default='gadgets/vcg_db.pkl')
    parser.add_argument('--vcg-res', default=None,
                        help='Path to vcg_circuit_resources.pkl for gate-count plots')
    parser.add_argument('--output-dir', default='analysis_output/figures/vcg_db/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading {args.db} ...')
    df = load_db_as_df(args.db)
    print_summary(df)

    print(f'\nGenerating plots → {args.output_dir}')
    plot_ar_by_type(df, args.output_dir)
    plot_entropy_by_type(df, args.output_dir)
    plot_layers_by_type(df, args.output_dir)
    plot_layers_vs_nx(df, args.output_dir)
    plot_entropy_vs_layers(df, args.output_dir)
    plot_train_time(df, args.output_dir)
    plot_vcg_layers_to_entropy_threshold(df, args.output_dir)
    plot_convergence_scatter(df, args.output_dir)

    if args.vcg_res and os.path.exists(args.vcg_res):
        import pandas as _pd
        res_df = _pd.read_pickle(args.vcg_res)
        plot_circuit_resources(res_df, args.output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
