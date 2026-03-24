"""
plot_vcg_db.py -- Generate VCG gadget-creation summary plots from noflag_db.pkl.

Plots produced
--------------
1. vcg_ar_by_type.png     — AR distribution by constraint family (all = 1.0)
2. vcg_entropy_by_type.png — H_norm (entropy) by constraint family, colour=n_x
3. vcg_layers_by_type.png  — Number of QAOA layers by constraint family
4. vcg_layers_vs_nx.png   — Layers distribution vs n_x (box + strip)
5. vcg_entropy_vs_layers.png — H_norm vs layers coloured by constraint family

Usage
-----
    python analyze_results/plot_vcg_db.py \
        --db gadgets/noflag_db.pkl \
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
            'constraint_key':   key,
            'constraint_type':  _classify(key),
            'n_x':              int(entry['n_x']),
            'ar':               float(entry['ar']),
            'entropy':          float(entry['entropy']) if entry['entropy'] is not None else np.nan,
            'n_layers':         int(entry['n_layers']),
            'train_time':       float(entry['train_time']) if entry.get('train_time') is not None else np.nan,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_LABELS = {
    'knapsack':           'Knapsack',
    'quadratic_knapsack': 'Quadratic knapsack',
}
NX_MARKERS = {3: 'o', 4: 's', 5: '^'}
NX_COLORS  = {3: _ROSE_PINE['pine'], 4: _ROSE_PINE['gold'], 5: _ROSE_PINE['iris']}


def plot_ar_by_type(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: mean AR per family. (Should all be 1.0.)"""
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
    """Strip + mean marker: H_norm per family, coloured by n_x."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    families = sorted(df['constraint_type'].unique())
    x_pos = {f: i for i, f in enumerate(families)}

    for nx in sorted(df['n_x'].unique()):
        sub = df[df['n_x'] == nx]
        for fam in families:
            vals = sub[sub['constraint_type'] == fam]['entropy'].dropna().values
            if len(vals) == 0:
                continue
            xi = x_pos[fam]
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals,
                       color=NX_COLORS[nx], marker=NX_MARKERS[nx],
                       s=40, alpha=0.75, zorder=3, label=f'$n_x={nx}$' if fam == families[0] else '')

    # Mean markers
    for fam in families:
        xi = x_pos[fam]
        mean_val = df[df['constraint_type'] == fam]['entropy'].mean()
        ax.hlines(mean_val, xi - 0.3, xi + 0.3, colors=_ROSE_PINE['text'],
                  linewidth=2, zorder=4)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in families])
    ax.set_ylabel('$H_{\\mathrm{norm}}$ (feasible entropy)')
    ax.set_title('VCG gadget $H_{\\mathrm{norm}}$ by constraint family')

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), title='$n_x$', framealpha=1)

    save_fig(fig, os.path.join(out_dir, 'vcg_entropy_by_type.png'))
    print(f'  Saved vcg_entropy_by_type.png')


def plot_layers_by_type(df: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart: layer count distribution per family."""
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


def plot_layers_vs_nx(df: pd.DataFrame, out_dir: str) -> None:
    """Box + strip: QAOA layers vs n_x."""
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    nx_vals = sorted(df['n_x'].unique())
    positions = list(range(len(nx_vals)))

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
        patch.set_alpha(0.5)

    rng = np.random.default_rng(7)
    for xi, (nx, pos) in enumerate(zip(nx_vals, positions)):
        vals = df[df['n_x'] == nx]['n_layers'].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(pos + jitter, vals,
                   color=NX_COLORS[nx], s=35, alpha=0.85, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'$n_x={nx}$' for nx in nx_vals])
    ax.set_ylabel('QAOA layers $p$')
    ax.set_title('VCG training depth vs problem size')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    save_fig(fig, os.path.join(out_dir, 'vcg_layers_vs_nx.png'))
    print(f'  Saved vcg_layers_vs_nx.png')


def plot_entropy_vs_layers(df: pd.DataFrame, out_dir: str) -> None:
    """Scatter: H_norm vs n_layers, coloured by constraint family."""
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
    ax.set_ylabel('$H_{\\mathrm{norm}}$')
    ax.set_title('Gadget entropy vs training depth')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()

    save_fig(fig, os.path.join(out_dir, 'vcg_entropy_vs_layers.png'))
    print(f'  Saved vcg_entropy_vs_layers.png')


def plot_train_time(df: pd.DataFrame, out_dir: str) -> None:
    """Strip + median marker: training time per family, coloured by n_x."""
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
                       label=f'$n_x={nx}$' if fam == families[0] else '')

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
    ax.legend(seen.values(), seen.keys(), title='$n_x$', framealpha=1)

    save_fig(fig, os.path.join(out_dir, 'vcg_train_time.png'))
    print(f'  Saved vcg_train_time.png')


def print_summary(df: pd.DataFrame) -> None:
    print('\n=== VCG Gadget DB Summary ===')
    print(f'Total gadgets: {len(df)}')
    print(f'\nBy constraint type:')
    print(df.groupby('constraint_type')[['ar', 'entropy', 'n_layers']].agg(
        ['count', 'mean', 'min', 'max']).to_string())
    print(f'\nBy n_x:')
    print(df.groupby('n_x')[['ar', 'entropy', 'n_layers']].agg(
        ['count', 'mean', 'min', 'max']).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot VCG gadget creation summary.')
    parser.add_argument('--db',         default='gadgets/noflag_db.pkl')
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

    print('\nDone.')


if __name__ == '__main__':
    main()
