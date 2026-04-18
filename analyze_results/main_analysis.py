"""
main_analysis.py -- CLI entry point for the constraint_gadget analysis pipeline.

Expects pre-split DataFrames produced by split_results.py.  Each input file
is small (no raw counts, no Hamiltonians, no opt_angles) so this script is
fast to run repeatedly with different plot options.

Usage
-----
    # After running split_results.py:
    python analyze_results/main_analysis.py \\
        --vcg-ar    results/vcg_ar.pkl \\
        --vcg-res   results/vcg_resources.pkl \\
        --comp-ar   results/comparison_ar.pkl \\
        --comp-res  results/comparison_resources.pkl \\
        --output-dir ./analysis_output/

    # VCG only:
    python analyze_results/main_analysis.py \\
        --vcg-ar results/vcg_ar.pkl --vcg-res results/vcg_resources.pkl

    # Hybrid comparison only:
    python analyze_results/main_analysis.py \\
        --comp-ar results/comparison_ar.pkl --comp-res results/comparison_resources.pkl

Output layout
-------------
    analysis_output/
        figures/ar/
        figures/feasibility/
        figures/resources/
        summaries/
        statistical_tests/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from analyze_results.metrics import summary_stats
from analyze_results import plot_ar, plot_feasibility, plot_resources
from analyze_results.statistical_tests import run_full_stats


def _makedirs(base: str) -> dict:
    """Create the standard output subdirectory tree under *base*.

    Returns a dict mapping short names to absolute paths::

        {
            'ar':          base/figures/ar/
            'feasibility': base/figures/feasibility/
            'resources':   base/figures/resources/
            'summaries':   base/summaries/
            'stats':       base/statistical_tests/
        }

    All directories are created with ``exist_ok=True``.
    """
    dirs = {
        'ar':          os.path.join(base, 'figures', 'ar'),
        'feasibility': os.path.join(base, 'figures', 'feasibility'),
        'resources':   os.path.join(base, 'figures', 'resources'),
        'summaries':   os.path.join(base, 'summaries'),
        'stats':       os.path.join(base, 'statistical_tests'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _load(path: str, label: str) -> pd.DataFrame:
    """Load a pickle file and print its row count.

    Parameters
    ----------
    path : str or None
        Path to the pickle file.  If ``None`` or the file does not exist,
        an empty DataFrame is returned silently.
    label : str
        Human-readable name printed alongside the row count (e.g. ``'VCG AR'``).

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame, or an empty DataFrame if the file is absent.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_pickle(path)
    print(f"  Loaded {label}: {len(df):,} rows from {os.path.basename(path)}")
    return df


def _generate_paper_stats(
    comp_ar_raw: pd.DataFrame,
    comp_ar: pd.DataFrame,
    comp_res: pd.DataFrame,
    vcg_ar: pd.DataFrame,
    n_total_raw: int,
    save_path: str,
) -> None:
    """Compute and write the key numbers cited in the results section.

    Writes a human-readable ``paper_stats.txt`` that is overwritten on every
    run so values always reflect the current dataset.

    Parameters
    ----------
    comp_ar_raw : unfiltered comparison AR DataFrame (all rows, including timeouts)
    comp_ar     : filtered comparison AR DataFrame (converged or p_max exhausted)
    comp_res    : comparison resources DataFrame
    vcg_ar      : VCG AR DataFrame
    n_total_raw : total rows in comp_ar before any filtering
    save_path   : path to write paper_stats.txt
    """
    import datetime
    lines = []
    lines.append(f"paper_stats.txt — generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # ------------------------------------------------------------------
    # HybridQAOA vs PenaltyQAOA
    # ------------------------------------------------------------------
    if not comp_ar.empty and 'method' in comp_ar.columns:
        lines.append("\n### Instance counts ###")
        for method, grp_raw in comp_ar_raw.groupby('method'):
            n_raw = len(grp_raw)
            grp_comp = comp_ar[comp_ar['method'] == method]
            n_comp = len(grp_comp)
            n_timeout = n_raw - n_comp
            lines.append(f"  {method}: {n_raw} total, {n_comp} completed, {n_timeout} timed out")

        p_max = int(comp_ar['n_layers'].max()) if 'n_layers' in comp_ar.columns else None

        lines.append("\n### Convergence (P(feas) >= 0.75) ###")
        for method, grp in comp_ar.groupby('method'):
            n = len(grp)
            if 'n_layers' in grp.columns and p_max is not None:
                n_converged_p1 = int(((grp['p_feasible'] >= 0.75) & (grp['n_layers'] == 1)).sum())
                n_exhausted    = int((grp['n_layers'] == p_max).sum())
                n_converged    = int((grp['p_feasible'] >= 0.75).sum())
                n_timeout      = len(comp_ar_raw[comp_ar_raw['method'] == method]) - n
                n_raw          = n + n_timeout
                pct_p1         = 100 * n_converged_p1 / n_raw
                pct_exhausted  = 100 * n_exhausted / n_raw
                pct_timeout    = 100 * n_timeout / n_raw
                pct_total_fail = 100 * (n_exhausted + n_timeout) / n_raw
                lines.append(f"  {method} (n={n_raw}):")
                lines.append(f"    converged at p=1:      {n_converged_p1:4d} ({pct_p1:.1f}%)")
                lines.append(f"    exhausted p_max={p_max}:   {n_exhausted:4d} ({pct_exhausted:.1f}%)")
                lines.append(f"    timed out:             {n_timeout:4d} ({pct_timeout:.1f}%)")
                lines.append(f"    total non-convergence: {n_exhausted + n_timeout:4d} ({pct_total_fail:.1f}%)")

        lines.append("\n### AR_feas ###")
        if 'AR_feas' not in comp_ar.columns:
            lines.append("  (AR_feas not in dataset)")
        else:
            for method, grp in comp_ar.groupby('method'):
                mean_all    = grp['AR_feas'].mean()
                median_all  = grp['AR_feas'].median()
                converged   = grp[grp['p_feasible'] >= 0.75]
                mean_conv   = converged['AR_feas'].mean() if len(converged) else float('nan')
                lines.append(f"  {method}: mean={mean_all:.3f}  median={median_all:.3f}  "
                             f"mean(converged only)={mean_conv:.3f}  n={len(grp)}")

        lines.append("\n### P(feas) by layer ###")
        if 'n_layers' in comp_ar.columns:
            for p in sorted(comp_ar['n_layers'].unique()):
                sub = comp_ar[comp_ar['n_layers'] == p]
                row = f"  p={p}:"
                for method, grp in sub.groupby('method'):
                    row += f"  {method} mean={grp['p_feasible'].mean():.3f} (n={len(grp)})"
                lines.append(row)

        lines.append("\n### P(feas) at p=1 per n_x ###")
        if 'n_layers' in comp_ar.columns and 'n_x' in comp_ar.columns:
            p1 = comp_ar[comp_ar['n_layers'] == 1]
            if not p1.empty:
                for nx, grp_nx in p1.groupby('n_x'):
                    row = f"  n_x={nx}:"
                    for method, grp in grp_nx.groupby('method'):
                        row += f"  {method}={grp['p_feasible'].mean():.3f}"
                    lines.append(row)

        lines.append("\n### P(opt) ###")
        if 'p_optimal' in comp_ar.columns:
            for method, grp in comp_ar.groupby('method'):
                mean_all = grp['p_optimal'].mean()
                if 'n_layers' in grp.columns:
                    p1_grp = grp[grp['n_layers'] == 1]
                    mean_p1 = p1_grp['p_optimal'].mean() if len(p1_grp) else float('nan')
                else:
                    mean_p1 = float('nan')
                lines.append(f"  {method}: mean(all)={mean_all:.3f}  mean(p=1)={mean_p1:.3f}  n={len(grp)}")

            methods = comp_ar['method'].unique()
            if len(methods) == 2 and 'n_layers' in comp_ar.columns:
                p1 = comp_ar[comp_ar['n_layers'] == 1]
                means = {m: p1[p1['method'] == m]['p_optimal'].mean() for m in methods}
                m0, m1 = methods[0], methods[1]
                if means[m1] > 0:
                    rel = 100 * (means[m0] - means[m1]) / means[m1]
                    lines.append(f"  Relative improvement {m0} vs {m1} at p=1: {rel:+.1f}%")

        lines.append("\n### P(opt) at p=1 per n_x ###")
        if 'p_optimal' in comp_ar.columns and 'n_layers' in comp_ar.columns and 'n_x' in comp_ar.columns:
            p1 = comp_ar[comp_ar['n_layers'] == 1]
            if not p1.empty:
                for nx, grp_nx in p1.groupby('n_x'):
                    row = f"  n_x={nx}:"
                    for method, grp in grp_nx.groupby('method'):
                        row += f"  {method}={grp['p_optimal'].mean():.3f}"
                    lines.append(row)

    # ------------------------------------------------------------------
    # Wall-clock time
    # ------------------------------------------------------------------
    if not comp_res.empty and 'method' in comp_res.columns and 'optimize_time' in comp_res.columns:
        lines.append("\n### Wall-clock time (optimize_time) ###")
        for method, grp in comp_res.groupby('method'):
            t_mean = grp['optimize_time'].mean()
            t_med  = grp['optimize_time'].median()
            lines.append(f"  {method}: mean={t_mean/3600:.2f}h  median={t_med/3600:.2f}h  n={len(grp)}")

        lines.append("\n### Wall-clock time per n_x ###")
        for nx, grp_nx in comp_res.groupby('n_x'):
            row = f"  n_x={nx}:"
            for method, grp in grp_nx.groupby('method'):
                row += f"  {method}={grp['optimize_time'].mean()/3600:.2f}h"
            lines.append(row)

    # ------------------------------------------------------------------
    # VCG
    # ------------------------------------------------------------------
    if not vcg_ar.empty:
        lines.append("\n### VCG gadget database ###")
        n_gadgets = len(vcg_ar)
        lines.append(f"  Total gadgets: {n_gadgets}")
        if 'n_layers' in vcg_ar.columns:
            n_p0 = int((vcg_ar['n_layers'] == 0).sum())
            lines.append(f"  Gadgets at p=0 (exact prep): {n_p0} ({100*n_p0/n_gadgets:.1f}%)")
            lines.append(f"  Max layers:    {int(vcg_ar['n_layers'].max())}")
            lines.append(f"  Mean layers:   {vcg_ar['n_layers'].mean():.2f}")
            lines.append(f"  Median layers: {vcg_ar['n_layers'].median():.1f}")
        if 'AR' in vcg_ar.columns:
            ar1 = (vcg_ar['AR'] >= 0.999).sum()
            lines.append(f"  Gadgets at AR=1.0: {ar1} ({100*ar1/n_gadgets:.1f}%)")
            lines.append(f"  Mean AR: {vcg_ar['AR'].mean():.4f}")
        if 'constraint_type' in vcg_ar.columns:
            lines.append("  By family:")
            for fam, grp in vcg_ar.groupby('constraint_type'):
                p0 = int((grp['n_layers'] == 0).sum()) if 'n_layers' in grp.columns else '?'
                lines.append(f"    {fam}: n={len(grp)}  p=0: {p0}  "
                             f"mean_layers={grp['n_layers'].mean():.1f}  AR={grp['AR'].mean():.4f}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    text = "\n".join(lines) + "\n"
    with open(save_path, 'w') as f:
        f.write(text)
    print(text)
    print(f"  → saved to {save_path}")


def _run_comparison_plots(comp_ar: pd.DataFrame, comp_ar_raw: pd.DataFrame,
                          comp_res: pd.DataFrame, dirs: dict) -> None:
    """Generate all comparison AR and feasibility plots into *dirs*."""
    if comp_ar.empty:
        return

    print("\n--- Comparison AR plots ---")

    for method in comp_ar.get('method', pd.Series(dtype=str)).unique():
        sub = comp_ar[comp_ar['method'] == method]
        plot_ar.plot_ar_by_n(
            sub, title=f'{method}: AR vs n_x',
            save_path=os.path.join(dirs['ar'], f'{method.lower()}_ar_by_n.png'))

    if 'method' in comp_ar.columns and comp_ar['method'].nunique() > 1:
        plot_ar.plot_ar_comparison(
            comp_ar,
            save_path=os.path.join(dirs['ar'], 'hybrid_vs_penalty_ar.png'))
        if 'AR_feas' in comp_ar.columns:
            plot_ar.plot_ar_feas_comparison(
                comp_ar,
                save_path=os.path.join(dirs['ar'], 'hybrid_vs_penalty_ar_feas.png'))

    plot_ar.plot_ar_vs_layers(
        comp_ar,
        save_path=os.path.join(dirs['ar'], 'ar_vs_layers.png'))

    if 'AR_feas' in comp_ar.columns:
        plot_ar.plot_ar_feas_vs_layers(
            comp_ar,
            save_path=os.path.join(dirs['ar'], 'ar_feas_vs_layers.png'))

    if 'layer' in comp_ar.columns and 'n_layers' in comp_ar.columns:
        plot_ar.plot_layers_to_threshold(
            comp_ar,
            save_path=os.path.join(dirs['ar'], 'layers_to_threshold.png'))

    if 'p_feasible' in comp_ar.columns:
        plot_feasibility.plot_p_feasible_hybrid(
            comp_ar,
            save_path=os.path.join(dirs['feasibility'], 'hybrid_p_feasible.png'))
        plot_feasibility.plot_p_feasible_comparison(
            comp_ar,
            save_path=os.path.join(dirs['feasibility'], 'hybrid_vs_penalty_p_feasible.png'))

    if 'p_optimal' in comp_ar.columns:
        plot_feasibility.plot_p_optimal_hybrid(
            comp_ar,
            save_path=os.path.join(dirs['feasibility'], 'hybrid_p_optimal.png'))

    groupby = [c for c in ['method', 'constraint_type', 'n_x', 'layer'] if c in comp_ar.columns]
    metrics = [c for c in ['AR', 'AR_feas', 'p_feasible', 'p_optimal'] if c in comp_ar.columns]
    if groupby and metrics:
        summary_stats(comp_ar, groupby, metrics).to_csv(
            os.path.join(dirs['summaries'], 'comparison_ar_summary.csv'), index=False)


def analyse(
    vcg_ar_path: str = None,
    vcg_res_path: str = None,
    comp_ar_path: str = None,
    comp_res_path: str = None,
    output_dir: str = './analysis_output/',
) -> None:
    """Generate all analysis plots and summary CSVs from pre-split result pickles.

    Expects pickles produced by ``split_results.py``.  Any path can be ``None``
    or missing — only the sections for which data is provided will run.

    For the hybrid vs penalty comparison, ``comparison_ar.pkl`` is expected to
    contain one row per experiment (the final-layer row, as produced by
    ``split_results.py``).  Experiments are further filtered to *completed*
    ones: either converged (``p_feasible >= 0.75``) or exhausted all ``p_max``
    layers.  Experiments cut short by cluster job limits are excluded so they
    do not dilute the metrics.

    Parameters
    ----------
    vcg_ar_path : str, optional
        Path to ``vcg_ar.pkl`` (VCG approximation-ratio data).
    vcg_res_path : str, optional
        Path to ``vcg_resources.pkl`` (VCG circuit-resource data).
    comp_ar_path : str, optional
        Path to ``comparison_ar.pkl`` (HybridQAOA vs PenaltyQAOA AR data).
    comp_res_path : str, optional
        Path to ``comparison_resources.pkl`` (HybridQAOA vs PenaltyQAOA resource data).
    output_dir : str
        Root directory for all outputs (figures, summaries, stats).
        Subdirectories are created automatically via :func:`_makedirs`.
    """
    dirs = _makedirs(output_dir)

    vcg_ar   = _load(vcg_ar_path,   'VCG AR')
    vcg_res  = _load(vcg_res_path,  'VCG Resources')
    comp_ar  = _load(comp_ar_path,  'Comparison AR')
    comp_res = _load(comp_res_path, 'Comparison Resources')

    # ------------------------------------------------------------------
    # VCG analysis
    # ------------------------------------------------------------------
    if not vcg_ar.empty:
        print("\n--- VCG plots ---")

        plot_ar.plot_ar_by_n(
            vcg_ar, title='VCG: AR vs n_x',
            save_path=os.path.join(dirs['ar'], 'vcg_ar_by_n.png'))
        plot_ar.plot_ar_by_constraint_type(
            vcg_ar,
            save_path=os.path.join(dirs['ar'], 'vcg_ar_by_constraint_type.png'))
        if 'angle_strategy' in vcg_ar.columns:
            plot_ar.plot_ar_by_angle_strategy(
                vcg_ar,
                save_path=os.path.join(dirs['ar'], 'vcg_ar_by_angle_strategy.png'))

        groupby = [c for c in ['constraint_type', 'n_x', 'angle_strategy'] if c in vcg_ar.columns]
        metrics = [c for c in ['AR', 'p_feasible'] if c in vcg_ar.columns]
        if groupby and metrics:
            summary_stats(vcg_ar, groupby, metrics).to_csv(
                os.path.join(dirs['summaries'], 'vcg_ar_summary.csv'), index=False)

    if not vcg_res.empty:
        print("\n--- VCG resource plots ---")

        if 'est_shots' in vcg_res.columns:
            plot_resources.plot_shots_vs_n(
                vcg_res,
                save_path=os.path.join(dirs['resources'], 'vcg_shots_vs_n.png'))
        if 'depth' in vcg_res.columns:
            plot_resources.plot_depth_vs_n(
                vcg_res,
                save_path=os.path.join(dirs['resources'], 'vcg_depth_vs_n.png'))
        plot_resources.plot_vcg_total_time(
            vcg_res,
            save_path=os.path.join(dirs['resources'], 'vcg_time_breakdown.png'))

        groupby = [c for c in ['constraint_type', 'n_x'] if c in vcg_res.columns]
        metrics = [c for c in ['est_shots', 'depth', 'optimize_time'] if c in vcg_res.columns]
        if groupby and metrics:
            summary_stats(vcg_res, groupby, metrics).to_csv(
                os.path.join(dirs['summaries'], 'vcg_resources_summary.csv'), index=False)

    # ------------------------------------------------------------------
    # Hybrid vs Penalty comparison analysis
    # ------------------------------------------------------------------
    comp_ar_raw = comp_ar.copy()
    n_total = len(comp_ar)
    if not comp_ar.empty:
        # Filter to completed experiments only: converged (p_feas >= 0.75) OR
        # exhausted all p_max layers without converging.  Experiments where the
        # job was killed before reaching either condition are excluded so
        # metrics are not diluted by incomplete results.
        # Note: comparison_ar.pkl already holds one row per experiment (the
        # final-layer row); this filter handles experiments cut short by
        # cluster job limits.
        p_max = int(comp_ar['n_layers'].max())
        completed = comp_ar[
            (comp_ar['p_feasible'] >= 0.75) |
            (comp_ar['n_layers'] == p_max)
        ].copy()
        n_total = len(comp_ar)
        n_comp  = len(completed)
        print(f"\n  Completed experiments: {n_comp}/{n_total} "
              f"({completed.groupby('method').size().to_dict()})")
        comp_ar = completed

        # All-instances plots (includes infeasible problems)
        _run_comparison_plots(comp_ar, comp_ar_raw, comp_res, dirs)

        # Feasible-instances-only plots (separate subdirectory)
        if 'has_feasible_solution' in comp_ar.columns:
            comp_ar_feas_only = comp_ar[comp_ar['has_feasible_solution']].copy()
            n_feas = len(comp_ar_feas_only)
            n_infeas = len(comp_ar) - n_feas
            print(f"\n  Feasible-only subset: {n_feas} rows "
                  f"({n_infeas} infeasible instances excluded)")
            if not comp_ar_feas_only.empty:
                feas_dirs = _makedirs(os.path.join(output_dir, 'feasible_only'))
                _run_comparison_plots(comp_ar_feas_only, comp_ar_raw, comp_res, feas_dirs)
                # Filter comp_res to the same feasible constraint sets
                if not comp_res.empty and 'constraints_hash' in comp_res.columns:
                    feasible_hashes = set(comp_ar_feas_only['constraints_hash'])
                    comp_res_feas_only = comp_res[
                        comp_res['constraints_hash'].isin(feasible_hashes)
                    ].copy()
                    if not comp_res_feas_only.empty:
                        plot_resources.plot_total_time_vs_nx(
                            comp_res_feas_only, comp_ar_feas_only,
                            save_path=os.path.join(feas_dirs['resources'],
                                                   'comparison_total_time.png'))

    if not comp_res.empty:
        print("\n--- Comparison resource plots ---")

        plot_resources.plot_shots_vs_n(
            comp_res,
            save_path=os.path.join(dirs['resources'], 'comparison_shots_vs_n.png'))
        plot_resources.plot_comparison_total_time(
            comp_res,
            save_path=os.path.join(dirs['resources'], 'comparison_time_breakdown.png'))
        plot_resources.plot_total_time_vs_nx(
            comp_res, comp_ar,
            save_path=os.path.join(dirs['resources'], 'comparison_total_time.png'))

        groupby = [c for c in ['method', 'constraint_type', 'n_x'] if c in comp_res.columns]
        metrics = [c for c in ['est_shots', 'optimize_time', 'num_gamma', 'num_beta']
                   if c in comp_res.columns]
        if groupby and metrics:
            summary_stats(comp_res, groupby, metrics).to_csv(
                os.path.join(dirs['summaries'], 'comparison_resources_summary.csv'), index=False)

    # ------------------------------------------------------------------
    # Statistical tests (uses AR splits for both VCG and hybrid)
    # ------------------------------------------------------------------
    print("\n--- Statistical tests ---")
    run_full_stats(vcg_ar, comp_ar, output_dir=dirs['stats'])

    # ------------------------------------------------------------------
    # Paper stats report
    # ------------------------------------------------------------------
    print("\n--- Paper stats ---")
    _generate_paper_stats(
        comp_ar_raw=comp_ar_raw,
        comp_ar=comp_ar,
        comp_res=comp_res,
        vcg_ar=vcg_ar,
        n_total_raw=n_total,
        save_path=os.path.join(dirs['summaries'], 'paper_stats.txt'),
    )

    print(f"\nAnalysis complete. Outputs in: {output_dir}")


def main() -> None:
    """CLI entry point.  Parse arguments and call :func:`analyse`."""
    parser = argparse.ArgumentParser(
        description='Analyse constraint_gadget experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--vcg-ar',   default=None,
                        help='VCG AR split pickle (from split_results.py)')
    parser.add_argument('--vcg-res',  default=None,
                        help='VCG resources split pickle')
    parser.add_argument('--comp-ar',  default=None,
                        help='Hybrid vs Penalty AR split pickle')
    parser.add_argument('--comp-res', default=None,
                        help='Hybrid vs Penalty resources split pickle')

    parser.add_argument('--output-dir', default='./analysis_output/',
                        help='Directory for all outputs (default: ./analysis_output/)')
    args = parser.parse_args()

    if not any([args.vcg_ar, args.vcg_res, args.comp_ar, args.comp_res]):
        parser.error('Provide at least one input file (--vcg-ar, --vcg-res, --comp-ar, --comp-res).')

    analyse(
        vcg_ar_path=args.vcg_ar,
        vcg_res_path=args.vcg_res,
        comp_ar_path=args.comp_ar,
        comp_res_path=args.comp_res,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
