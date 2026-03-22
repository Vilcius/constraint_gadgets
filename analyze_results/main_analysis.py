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
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_pickle(path)
    print(f"  Loaded {label}: {len(df):,} rows from {os.path.basename(path)}")
    return df


def analyse(
    vcg_ar_path: str = None,
    vcg_res_path: str = None,
    comp_ar_path: str = None,
    comp_res_path: str = None,
    output_dir: str = './analysis_output/',
) -> None:
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
        plot_ar.plot_ar_by_angle_strategy(
            vcg_ar,
            save_path=os.path.join(dirs['ar'], 'vcg_ar_by_angle_strategy.png'))

        if 'p_feasible' in vcg_ar.columns:
            plot_feasibility.plot_p_feasible_vcg(
                vcg_ar,
                save_path=os.path.join(dirs['feasibility'], 'vcg_p_feasible.png'))
            plot_feasibility.plot_ar_vs_p_feasible(
                vcg_ar,
                save_path=os.path.join(dirs['feasibility'], 'vcg_ar_vs_p_feasible.png'))

        groupby = [c for c in ['constraint_type', 'n_x', 'angle_strategy'] if c in vcg_ar.columns]
        metrics = [c for c in ['AR', 'p_feasible'] if c in vcg_ar.columns]
        if groupby and metrics:
            summary_stats(vcg_ar, groupby, metrics).to_csv(
                os.path.join(dirs['summaries'], 'vcg_ar_summary.csv'), index=False)

    if not vcg_res.empty:
        print("\n--- VCG resource plots ---")

        plot_resources.plot_shots_vs_n(
            vcg_res,
            save_path=os.path.join(dirs['resources'], 'vcg_shots_vs_n.png'))
        plot_resources.plot_depth_vs_n(
            vcg_res,
            save_path=os.path.join(dirs['resources'], 'vcg_depth_vs_n.png'))
        plot_resources.plot_time_breakdown(
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
    if not comp_ar.empty:
        print("\n--- Comparison AR plots ---")

        # AR by n_x, separated by method
        for method in comp_ar.get('method', pd.Series(dtype=str)).unique():
            sub = comp_ar[comp_ar['method'] == method]
            plot_ar.plot_ar_by_n(
                sub, title=f'{method}: AR vs n_x',
                save_path=os.path.join(dirs['ar'], f'{method.lower()}_ar_by_n.png'))

        # Hybrid vs Penalty head-to-head AR comparison
        if 'method' in comp_ar.columns and comp_ar['method'].nunique() > 1:
            plot_ar.plot_ar_comparison(
                comp_ar,
                save_path=os.path.join(dirs['ar'], 'hybrid_vs_penalty_ar.png'))
            if 'AR_feas' in comp_ar.columns:
                plot_ar.plot_ar_feas_comparison(
                    comp_ar,
                    save_path=os.path.join(dirs['ar'], 'hybrid_vs_penalty_ar_feas.png'))

        # AR vs QAOA layers
        plot_ar.plot_ar_vs_layers(
            comp_ar,
            save_path=os.path.join(dirs['ar'], 'ar_vs_layers.png'))

        # Feasibility
        if 'p_feasible' in comp_ar.columns:
            plot_feasibility.plot_p_feasible_hybrid(
                comp_ar,
                save_path=os.path.join(dirs['feasibility'], 'hybrid_p_feasible.png'))
            plot_feasibility.plot_p_feasible_comparison(
                comp_ar,
                save_path=os.path.join(dirs['feasibility'], 'hybrid_vs_penalty_p_feasible.png'))

        # Optimality
        if 'p_optimal' in comp_ar.columns:
            plot_feasibility.plot_p_optimal_hybrid(
                comp_ar,
                save_path=os.path.join(dirs['feasibility'], 'hybrid_p_optimal.png'))

        # Summaries
        groupby = [c for c in ['method', 'constraint_type', 'n_x', 'layer'] if c in comp_ar.columns]
        metrics = [c for c in ['AR', 'AR_feas', 'p_feasible', 'p_optimal'] if c in comp_ar.columns]
        if groupby and metrics:
            summary_stats(comp_ar, groupby, metrics).to_csv(
                os.path.join(dirs['summaries'], 'comparison_ar_summary.csv'), index=False)

    if not comp_res.empty:
        print("\n--- Comparison resource plots ---")

        plot_resources.plot_shots_vs_n(
            comp_res,
            save_path=os.path.join(dirs['resources'], 'comparison_shots_vs_n.png'))
        plot_resources.plot_time_breakdown(
            comp_res,
            save_path=os.path.join(dirs['resources'], 'comparison_time_breakdown.png'))

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

    print(f"\nAnalysis complete. Outputs in: {output_dir}")


def main() -> None:
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
