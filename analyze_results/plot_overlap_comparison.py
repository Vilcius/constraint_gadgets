"""
plot_overlap_comparison.py -- Box-and-whisker plot comparing HybridQAOA
on disjoint (fully structural) vs overlapping constraint tasks.

Usage
-----
    python analyze_results/plot_overlap_comparison.py \
        --disjoint  results/archive_disjoint/pending/ \
        --overlap   results/pending/ \
        --output    analysis_output/figures/feasibility/hybrid_disjoint_vs_overlap.png
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyze_results import plot_utils as pu
from analyze_results.metrics import p_feasible_hybrid, p_optimal_hybrid


def _load_pending(pending_dir: str) -> pd.DataFrame:
    """Load all ``task_*.pkl`` files from *pending_dir* into a single DataFrame.

    After concatenation, any column whose every value is a length-1 list is
    unwrapped so downstream code can treat values as scalars.  Files that
    fail to load are skipped silently.

    Parameters
    ----------
    pending_dir : str
        Directory containing per-task pickle files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame, or an empty DataFrame if no valid files found.
    """
    files = sorted(glob.glob(os.path.join(pending_dir, 'task_*.pkl')))
    frames = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            if isinstance(df, pd.DataFrame):
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Unpack length-1 list columns
    for col in df.columns:
        try:
            if df[col].apply(lambda v: isinstance(v, list) and len(v) == 1).all():
                df[col] = df[col].apply(lambda v: v[0])
        except Exception:
            pass
    return df


def _ensure_constraints_col(df: pd.DataFrame) -> pd.DataFrame:
    """New-format data stores constraints inside the 'task' dict."""
    if 'constraints' not in df.columns or df['constraints'].isna().all():
        if 'task' in df.columns:
            def _extract(t):
                if isinstance(t, dict):
                    return t.get('constraints', [])
                return []
            df = df.copy()
            df['constraints'] = df['task'].apply(_extract)
    return df


def plot_disjoint_vs_overlap(
    disjoint_dir: str,
    overlap_dir: str,
    save_path: str = None,
) -> plt.Figure:
    """Side-by-side box plots of p_feas and p_opt for disjoint vs overlapping tasks."""

    raw_dis = _load_pending(disjoint_dir)
    raw_ov  = _load_pending(overlap_dir)

    for df in (raw_dis, raw_ov):
        if df.empty:
            raise ValueError(f"No data loaded from one of the directories.")

    raw_dis = _ensure_constraints_col(raw_dis)
    raw_ov  = _ensure_constraints_col(raw_ov)

    # Keep only HybridQAOA, layer=1
    def _filter(df):
        method = df['method'].apply(lambda x: x if isinstance(x, str) else (x[0] if x else ''))
        nl     = df['n_layers'].apply(lambda x: x if not isinstance(x, list) else x[0])
        return df[(method == 'HybridQAOA') & (nl == 1)].copy()

    dis = _filter(raw_dis)
    ov  = _filter(raw_ov)

    dis['p_feas'] = dis.apply(p_feasible_hybrid, axis=1)
    dis['p_opt']  = dis.apply(p_optimal_hybrid,  axis=1)
    ov['p_feas']  = ov.apply(p_feasible_hybrid,  axis=1)
    ov['p_opt']   = ov.apply(p_optimal_hybrid,   axis=1)

    dis = dis.dropna(subset=['p_feas', 'p_opt'])
    ov  = ov.dropna(subset=['p_feas', 'p_opt'])

    pu.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    color_dis = pu._ROSE_PINE['pine']    # blue for disjoint
    color_ov  = pu._ROSE_PINE['iris']    # purple for overlapping

    def _boxplot(ax, data_dis, data_ov, ylabel, title):
        bp = ax.boxplot(
            [data_dis.dropna(), data_ov.dropna()],
            labels=['Disjoint', 'Overlapping'],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color=pu._ROSE_PINE['text'], linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=3, alpha=0.5),
        )
        for patch, color in zip(bp['boxes'], [color_dis, color_ov]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for flier, color in zip(bp['fliers'], [color_dis, color_ov]):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)

        # Annotate medians
        for i, data in enumerate([data_dis, data_ov], start=1):
            med = np.nanmedian(data)
            mean = np.nanmean(data)
            ax.text(i, med + 0.02, f'med={med:.2f}', ha='center', va='bottom',
                    fontsize=9, color=pu._ROSE_PINE['text'])
            ax.text(i, -0.06, f'n={len(data.dropna())}', ha='center', va='top',
                    fontsize=8, color=pu._ROSE_PINE['muted'],
                    transform=ax.get_xaxis_transform())

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.12)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    _boxplot(axes[0], dis['p_feas'], ov['p_feas'],
             r'$P(\mathrm{feasible})$', r'$P(\mathrm{feasible})$: HybridQAOA, $p=1$')
    _boxplot(axes[1], dis['p_opt'],  ov['p_opt'],
             r'$P(\mathrm{optimal})$',  r'$P(\mathrm{optimal})$: HybridQAOA, $p=1$')

    if save_path:
        pu.save_fig(fig, save_path)
    return fig


def main() -> None:
    """CLI entry point.  Parse arguments and call :func:`plot_disjoint_vs_overlap`."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--disjoint', default='results/archive_disjoint/pending/',
                        help='Pending dir for disjoint (archive) results')
    parser.add_argument('--overlap',  default='results/pending/',
                        help='Pending dir for overlapping results')
    parser.add_argument('--output',
                        default='analysis_output/figures/feasibility/hybrid_disjoint_vs_overlap.png',
                        help='Output PNG path')
    args = parser.parse_args()

    fig = plot_disjoint_vs_overlap(args.disjoint, args.overlap, save_path=args.output)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
