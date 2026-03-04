"""
main_analysis.py -- CLI entry point for the constraint_gadget analysis pipeline.

Usage
-----
    python analyze_results/main_analysis.py \\
        --vcg   results/cardinality_constraint_results.pkl \\
        --hybrid results/hybrid_cardinality_results.pkl \\
        --output-dir ./analysis_output/

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

# Allow running from project root: python analyze_results/main_analysis.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from analyze_results.data_loader import load_vcg_results, load_hybrid_results
from analyze_results.metrics import add_vcg_metrics, add_hybrid_metrics, summary_stats
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


def analyse(vcg_path: str = None, hybrid_path: str = None,
            output_dir: str = './analysis_output/') -> None:
    dirs = _makedirs(output_dir)

    vcg_df    = pd.DataFrame()
    hybrid_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # VCG results
    # ------------------------------------------------------------------
    if vcg_path:
        print(f"Loading VCG results from {vcg_path} ...")
        vcg_df = load_vcg_results(vcg_path)
        vcg_df = add_vcg_metrics(vcg_df)
        print(f"  {len(vcg_df)} rows loaded.")

        # AR plots
        plot_ar.plot_ar_by_n(
            vcg_df, title='VCG: AR vs n_x',
            save_path=os.path.join(dirs['ar'], 'vcg_ar_by_n.png'))
        plot_ar.plot_ar_by_constraint_type(
            vcg_df,
            save_path=os.path.join(dirs['ar'], 'vcg_ar_by_constraint_type.png'))
        plot_ar.plot_ar_by_angle_strategy(
            vcg_df,
            save_path=os.path.join(dirs['ar'], 'vcg_ar_by_angle_strategy.png'))

        # Feasibility plots
        if 'p_feasible' in vcg_df.columns:
            plot_feasibility.plot_p_feasible_vcg(
                vcg_df,
                save_path=os.path.join(dirs['feasibility'], 'vcg_p_feasible.png'))
            plot_feasibility.plot_ar_vs_p_feasible(
                vcg_df,
                save_path=os.path.join(dirs['feasibility'], 'vcg_ar_vs_p_feasible.png'))

        # Resource plots
        plot_resources.plot_shots_vs_n(
            vcg_df,
            save_path=os.path.join(dirs['resources'], 'vcg_shots_vs_n.png'))
        plot_resources.plot_depth_vs_n(
            vcg_df,
            save_path=os.path.join(dirs['resources'], 'vcg_depth_vs_n.png'))
        plot_resources.plot_time_breakdown(
            vcg_df,
            save_path=os.path.join(dirs['resources'], 'vcg_time_breakdown.png'))

        # Summaries
        metrics_vcg = [c for c in ['AR', 'p_feasible', 'depth', 'num_gates']
                       if c in vcg_df.columns]
        groupby_vcg = [c for c in ['constraint_type', 'n_x', 'angle_strategy']
                       if c in vcg_df.columns]
        if groupby_vcg and metrics_vcg:
            summary = summary_stats(vcg_df, groupby_vcg, metrics_vcg)
            summary.to_csv(os.path.join(dirs['summaries'], 'vcg_summary.csv'), index=False)

    # ------------------------------------------------------------------
    # Hybrid results
    # ------------------------------------------------------------------
    if hybrid_path:
        print(f"Loading Hybrid results from {hybrid_path} ...")
        hybrid_df = load_hybrid_results(hybrid_path)
        hybrid_df = add_hybrid_metrics(hybrid_df)
        print(f"  {len(hybrid_df)} rows loaded.")

        # AR plots
        plot_ar.plot_ar_by_n(
            hybrid_df, title='HybridQAOA: AR vs n_x',
            save_path=os.path.join(dirs['ar'], 'hybrid_ar_by_n.png'))

        # Feasibility / optimality
        if 'p_feasible' in hybrid_df.columns:
            plot_feasibility.plot_p_feasible_hybrid(
                hybrid_df,
                save_path=os.path.join(dirs['feasibility'], 'hybrid_p_feasible.png'))
        if 'p_optimal' in hybrid_df.columns:
            plot_feasibility.plot_p_optimal_hybrid(
                hybrid_df,
                save_path=os.path.join(dirs['feasibility'], 'hybrid_p_optimal.png'))

        # Resources
        plot_resources.plot_shots_vs_n(
            hybrid_df,
            save_path=os.path.join(dirs['resources'], 'hybrid_shots_vs_n.png'))
        plot_resources.plot_time_breakdown(
            hybrid_df,
            save_path=os.path.join(dirs['resources'], 'hybrid_time_breakdown.png'))

        # Summaries
        metrics_hyb = [c for c in ['AR', 'p_feasible', 'p_optimal']
                       if c in hybrid_df.columns]
        groupby_hyb = [c for c in ['constraint_type', 'n_x', 'mixer']
                       if c in hybrid_df.columns]
        if groupby_hyb and metrics_hyb:
            summary = summary_stats(hybrid_df, groupby_hyb, metrics_hyb)
            summary.to_csv(os.path.join(dirs['summaries'], 'hybrid_summary.csv'), index=False)

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------
    run_full_stats(vcg_df, hybrid_df, output_dir=dirs['stats'])
    print(f"\nAnalysis complete. Outputs in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyse constraint_gadget experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--vcg',    type=str, default=None,
                        help='Path (or glob) to VCG result pickle(s)')
    parser.add_argument('--hybrid', type=str, default=None,
                        help='Path (or glob) to HybridQAOA result pickle(s)')
    parser.add_argument('--output-dir', type=str, default='./analysis_output/',
                        help='Directory for all outputs (default: ./analysis_output/)')
    args = parser.parse_args()

    if args.vcg is None and args.hybrid is None:
        parser.error('Provide at least one of --vcg or --hybrid.')

    analyse(vcg_path=args.vcg, hybrid_path=args.hybrid, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
