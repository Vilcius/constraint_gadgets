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

    # PC-QAOA comparison only:
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
from analyze_results.plot_feasibility import add_vcg_flag
from analyze_results.statistical_tests import run_full_stats
from analyze_results.plot_vcg_db import (
    load_db_as_df, print_summary,
    plot_ar_by_type, plot_entropy_by_type, plot_layers_by_type,
    plot_layers_vs_nx, plot_entropy_vs_layers, plot_train_time,
    plot_circuit_resources, plot_vcg_layers_to_entropy_threshold,
    plot_convergence_scatter,
)


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
        'ar': os.path.join(base, 'figures', 'ar'),
        'feasibility': os.path.join(base, 'figures', 'feasibility'),
        'resources': os.path.join(base, 'figures', 'resources'),
        'vcg_db': os.path.join(base, 'figures', 'vcg_db'),
        'summaries': os.path.join(base, 'summaries'),
        'stats': os.path.join(base, 'statistical_tests'),
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


def _generate_vcg_paper_stats(vcg_df: pd.DataFrame, save_path: str) -> None:
    """Compute and write VCG gadget-database stats to *save_path*.

    Covers exact vs QAOA breakdown, entropy/AR/convergence by family and
    support size, n_feasible, and training time.  All numbers exclude
    exact-preparation gadgets unless noted.
    """
    import datetime
    lines = []
    lines.append(f"vcg_paper_stats.txt — generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    total = len(vcg_df)
    n_exact = int(vcg_df['is_exact'].sum())
    n_qaoa = total - n_exact
    lines.append(f"\n### Gadget database overview ###")
    lines.append(f"  Total gadgets:           {total}")
    lines.append(f"  Exact-prep (p=0):        {n_exact}  ({100*n_exact/total:.1f}%)")
    lines.append(f"  QAOA-trained:            {n_qaoa}  ({100*n_qaoa/total:.1f}%)")

    lines.append(f"\n  By family (all):")
    for fam, grp in vcg_df.groupby('constraint_type'):
        exact = int(grp['is_exact'].sum())
        lines.append(f"    {fam}: total={len(grp)}  exact={exact}  QAOA={len(grp)-exact}")

    qaoa = vcg_df[~vcg_df['is_exact']].copy()

    # ── Convergence ───────────────────────────────────────────────────────
    lines.append(f"\n### Convergence (QAOA gadgets only) ###")
    lines.append(f"  Criteria: AR >= 0.999 AND S_norm >= 0.9999")
    if 'converged' in qaoa.columns:
        n_conv = int(qaoa['converged'].sum())
        n_notconv = n_qaoa - n_conv
        lines.append(f"  Converged early:         {n_conv}/{n_qaoa}  ({100*n_conv/n_qaoa:.1f}%)")
        lines.append(f"  Ran to p_max=8:          {n_notconv}/{n_qaoa}  ({100*n_notconv/n_qaoa:.1f}%)")
        lines.append(f"\n  By family:")
        for fam, grp in qaoa.groupby('constraint_type'):
            conv = int(grp['converged'].sum())
            lines.append(f"    {fam}: converged={conv}/{len(grp)}")

    # ── Entropy ──────────────────────────────────────────────────────────
    lines.append(f"\n### Normalized feasible entropy S_norm (QAOA only) ###")
    for fam, grp in qaoa.groupby('constraint_type'):
        e = grp['entropy'].dropna()
        lines.append(f"  {fam}:  n={len(grp)}  mean={e.mean():.4f}  std={e.std():.4f}"
                     f"  min={e.min():.4f}  max={e.max():.4f}")
        for thresh in [0.9999, 0.999, 0.99, 0.9, 0.75]:
            cnt = int((e >= thresh).sum())
            lines.append(f"    >= {thresh:.4f}: {cnt}/{len(grp)}  ({100*cnt/len(grp):.1f}%)")

    lines.append(f"\n  By family and |S_k|:")
    for (fam, nx), grp in qaoa.groupby(['constraint_type', 'n_x']):
        e = grp['entropy'].dropna()
        conv = int(grp['converged'].sum()) if 'converged' in grp.columns else '?'
        lines.append(f"    {fam} |S_k|={nx}: n={len(grp)}  mean={e.mean():.4f}"
                     f"  std={e.std():.4f}  converged={conv}/{len(grp)}")

    # ── AR ────────────────────────────────────────────────────────────────
    lines.append(f"\n### Approximation ratio AR (QAOA only) ###")
    for fam, grp in qaoa.groupby('constraint_type'):
        a = grp['ar'].dropna()
        lines.append(f"  {fam}:  mean={a.mean():.4f}  min={a.min():.4f}")
        for thresh in [0.999, 0.99, 0.9]:
            cnt = int((a >= thresh).sum())
            lines.append(f"    >= {thresh}: {cnt}/{len(grp)}  ({100*cnt/len(grp):.1f}%)")

    # ── Layers ───────────────────────────────────────────────────────────
    lines.append(f"\n### QAOA depth at best-entropy layer (QAOA only) ###")
    for fam, grp in qaoa.groupby('constraint_type'):
        lyr = grp['n_layers'].dropna()
        lines.append(f"  {fam}:  mean={lyr.mean():.2f}  median={lyr.median():.1f}"
                     f"  min={int(lyr.min())}  max={int(lyr.max())}")
    lines.append(f"\n  By family and |S_k|:")
    for (fam, nx), grp in qaoa.groupby(['constraint_type', 'n_x']):
        lyr = grp['n_layers'].dropna()
        lines.append(f"    {fam} |S_k|={nx}: mean_layers={lyr.mean():.1f}"
                     f"  min={int(lyr.min())}  max={int(lyr.max())}")

    # ── n_feasible ───────────────────────────────────────────────────────
    if 'n_feasible' in qaoa.columns:
        lines.append(f"\n### Feasible set size n_feasible (QAOA only) ###")
        for fam, grp in qaoa.groupby('constraint_type'):
            nf = grp['n_feasible'].dropna()
            lines.append(f"  {fam}:  mean={nf.mean():.1f}  min={int(nf.min())}  max={int(nf.max())}")
        lines.append(f"\n  By family and |S_k|:")
        for (fam, nx), grp in qaoa.groupby(['constraint_type', 'n_x']):
            nf = grp['n_feasible'].dropna()
            lines.append(f"    {fam} |S_k|={nx}: mean={nf.mean():.1f}"
                         f"  min={int(nf.min())}  max={int(nf.max())}")

    # ── Training time ────────────────────────────────────────────────────
    if 'train_time' in qaoa.columns:
        lines.append(f"\n### Training time (QAOA only) ###")
        for fam, grp in qaoa.groupby('constraint_type'):
            t = grp['train_time'].dropna()
            lines.append(f"  {fam}:  mean={t.mean():.1f}s  total={t.sum():.0f}s")
        total_t = qaoa['train_time'].dropna().sum()
        lines.append(f"  All QAOA gadgets total: {total_t:.0f}s ({total_t/3600:.2f}h)")

    text = "\n".join(lines) + "\n"
    with open(save_path, 'w') as f:
        f.write(text)
    print(text)
    print(f"  → saved to {save_path}")


def _generate_paper_stats(
    comp_ar_raw: pd.DataFrame,
    comp_ar: pd.DataFrame,
    comp_res: pd.DataFrame,
    vcg_ar: pd.DataFrame,
    n_total_raw: int,
    save_path: str,
    comp_ar_conv: pd.DataFrame = None,
    comp_ar_all: pd.DataFrame = None,
    exact_lookup: dict = None,
    circ_df: pd.DataFrame = None,
) -> None:
    """Compute and write the key numbers cited in the results section.

    Writes a human-readable ``paper_stats.txt`` that is overwritten on every
    run so values always reflect the current dataset.

    Parameters
    ----------
    comp_ar_raw  : unfiltered comparison AR DataFrame (all rows, including timeouts)
    comp_ar      : filtered comparison AR DataFrame (converged or p_max exhausted)
    comp_res     : comparison resources DataFrame
    vcg_ar       : VCG AR DataFrame
    n_total_raw  : total rows in comp_ar before any filtering
    save_path    : path to write paper_stats.txt
    comp_ar_conv : convergence-layer DataFrame (one row per experiment)
    comp_ar_all  : all-layers DataFrame (one row per experiment per layer)
    exact_lookup : {constraint_key: is_exact} from VCG DB
    """
    import datetime
    lines = []
    lines.append(f"paper_stats.txt — generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # Use all-layers data for per-layer stats; fall back to final-layer data.
    all_layers = comp_ar_all if (comp_ar_all is not None and not comp_ar_all.empty) else comp_ar

    def _at_layer(df, layer):
        if 'layer' in df.columns:
            return df[df['layer'] == layer]
        if 'n_layers' in df.columns:
            return df[df['n_layers'] == layer]
        return pd.DataFrame()

    def _metric_rows(df, metric, p_max, label_prefix=''):
        """Emit p=1 and p=p_max rows for a metric, grouped by method."""
        rows = []
        for p_label, lyr in [('p=1', 1), (f'p={p_max}', p_max)]:
            src = _at_layer(df, lyr)
            if src.empty or metric not in src.columns:
                continue
            row = f"  {label_prefix}{p_label}:"
            for method, grp in src.groupby('method'):
                v = grp[metric].mean()
                if metric == 'AR_feas':
                    row += f"  {method} mean={v:.3f}  median={grp[metric].median():.3f}"
                else:
                    row += f"  {method} mean={v:.3f}"
            rows.append(row)
        return rows

    def _per_n_rows(df, metric, lyr, label_prefix='    '):
        """Emit one row per n_x showing metric mean by method."""
        rows = []
        src = _at_layer(df, lyr)
        if src.empty or metric not in src.columns or 'n_x' not in src.columns:
            return rows
        for nx, grp_nx in src.groupby('n_x'):
            row = f"{label_prefix}n={nx}:"
            for method, grp in grp_nx.groupby('method'):
                row += f"  {method}={grp[metric].mean():.3f}"
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # PC-QAOA vs PenaltyQAOA
    # ------------------------------------------------------------------
    if not comp_ar.empty and 'method' in comp_ar.columns:
        p_max = int(comp_ar['layer'].max()) if 'layer' in comp_ar.columns else (
            int(comp_ar['n_layers'].max()) if 'n_layers' in comp_ar.columns else 5)

        has_splits = 'overlap_type' in all_layers.columns
        splits = sorted(all_layers['overlap_type'].unique()) if has_splits else []

        # ── Instance counts ───────────────────────────────────────────
        lines.append("\n### Instance counts ###")
        lines.append("  (one instance = one problem solved at all p layers)")
        for method, grp_raw in comp_ar_raw.groupby('method'):
            n_raw = len(grp_raw['constraints_hash'].unique()) if 'constraints_hash' in grp_raw.columns else len(grp_raw)
            grp_comp = comp_ar[comp_ar['method'] == method]
            n_comp = len(grp_comp['constraints_hash'].unique()) if 'constraints_hash' in grp_comp.columns else len(grp_comp)
            n_timeout = n_raw - n_comp
            lines.append(f"  {method}: {n_raw} instances, {n_comp} completed, {n_timeout} timed out")
        if has_splits:
            lines.append(f"\n  By overlap type (all experiment-rows in all_layers):")
            for sp in splits:
                sub = all_layers[all_layers['overlap_type'] == sp]
                lines.append(f"    {sp}: {sub['constraints_hash'].nunique() if 'constraints_hash' in sub.columns else len(sub)//p_max} instances")

        # ── Convergence ───────────────────────────────────────────────
        conv_src = comp_ar_conv if (comp_ar_conv is not None and not comp_ar_conv.empty) else comp_ar
        lines.append("\n### Convergence (P(feas) >= 0.75 OR p_max exhausted) ###")
        for method, grp in conv_src.groupby('method'):
            n = len(grp)
            if 'n_layers' not in grp.columns:
                continue
            n_converged_p1 = int((grp['n_layers'] == 1).sum())
            n_exhausted = int((grp['n_layers'] == p_max).sum())
            n_timeout = len(comp_ar_raw[comp_ar_raw['method'] == method]) - n
            n_raw = n + n_timeout
            lines.append(f"  {method} (n={n_raw}):")
            lines.append(f"    converged at p=1:      {n_converged_p1:4d} ({100*n_converged_p1/n_raw:.1f}%)")
            lines.append(f"    exhausted p_max={p_max}:   {n_exhausted:4d} ({100*n_exhausted/n_raw:.1f}%)")
            lines.append(f"    timed out:             {n_timeout:4d} ({100*n_timeout/n_raw:.1f}%)")
            lines.append(f"    total non-convergence: {n_exhausted+n_timeout:4d} ({100*(n_exhausted+n_timeout)/n_raw:.1f}%)")

        # ── AR_feas ───────────────────────────────────────────────────
        if 'AR_feas' in all_layers.columns:
            lines.append("\n### AR_feas (approximation ratio over feasible solutions) ###")
            lines.extend(_metric_rows(all_layers, 'AR_feas', p_max))
            for p_lyr in [1, p_max]:
                lines.append(f"\n  Per n  (p={p_lyr}):")
                lines.extend(_per_n_rows(all_layers, 'AR_feas', p_lyr))
            if has_splits:
                for sp in splits:
                    sub = all_layers[all_layers['overlap_type'] == sp]
                    lines.append(f"\n  [{sp}]")
                    lines.extend(_metric_rows(sub, 'AR_feas', p_max, label_prefix='  '))

        # ── P(feas) ───────────────────────────────────────────────────
        if 'p_feasible' in all_layers.columns:
            lines.append("\n### P(feas) ###")
            lines.extend(_metric_rows(all_layers, 'p_feasible', p_max))
            for p_lyr in [1, p_max]:
                lines.append(f"\n  Per n  (p={p_lyr}):")
                lines.extend(_per_n_rows(all_layers, 'p_feasible', p_lyr))
            if has_splits:
                for sp in splits:
                    sub = all_layers[all_layers['overlap_type'] == sp]
                    lines.append(f"\n  [{sp}]")
                    lines.extend(_metric_rows(sub, 'p_feasible', p_max, label_prefix='  '))

        # ── P(opt) ────────────────────────────────────────────────────
        if 'p_optimal' in all_layers.columns:
            lines.append("\n### P(opt) ###")
            lines.extend(_metric_rows(all_layers, 'p_optimal', p_max))
            # Relative improvement at p=1
            src1 = _at_layer(all_layers, 1)
            if not src1.empty:
                methods_list = [m for m in ['PC-QAOA', 'PenaltyQAOA'] if m in src1['method'].values]
                if len(methods_list) == 2:
                    m0, m1 = methods_list
                    v0 = src1[src1['method'] == m0]['p_optimal'].mean()
                    v1 = src1[src1['method'] == m1]['p_optimal'].mean()
                    if v1 > 0:
                        lines.append(f"  Relative improvement {m0} vs {m1} at p=1: {100*(v0-v1)/v1:+.1f}%")
            for p_lyr in [1, p_max]:
                lines.append(f"\n  Per n  (p={p_lyr}):")
                lines.extend(_per_n_rows(all_layers, 'p_optimal', p_lyr))
            if has_splits:
                for sp in splits:
                    sub = all_layers[all_layers['overlap_type'] == sp]
                    lines.append(f"\n  [{sp}]")
                    lines.extend(_metric_rows(sub, 'p_optimal', p_max, label_prefix='  '))

    # ------------------------------------------------------------------
    # Wall-clock time
    # ------------------------------------------------------------------
    if not comp_res.empty and 'method' in comp_res.columns and 'optimize_time' in comp_res.columns:
        lines.append("\n### Wall-clock time (optimize_time) ###")
        for method, grp in comp_res.groupby('method'):
            t_mean = grp['optimize_time'].mean()
            t_med = grp['optimize_time'].median()
            lines.append(f"  {method}: mean={t_mean/3600:.2f}h  median={t_med/3600:.2f}h  n={len(grp)}")
        lines.append("\n  Per n:")
        for nx, grp_nx in comp_res.groupby('n_x'):
            row = f"    n={nx}:"
            for method, grp in grp_nx.groupby('method'):
                row += f"  {method}={grp['optimize_time'].mean()/3600:.2f}h"
            lines.append(row)

    # ------------------------------------------------------------------
    # VCG (QAOA-trained) vs exact prep breakdown
    # ------------------------------------------------------------------
    if exact_lookup is not None and not comp_ar.empty and 'structural_constraints' in comp_ar.columns:
        from core.constraint_handler import normalize_constraint as _nc
        all_layers_vc = comp_ar_all if (comp_ar_all is not None and not comp_ar_all.empty) else comp_ar

        def _prep(row):
            for c in (row['structural_constraints'] or []):
                if _nc(c) in exact_lookup and not exact_lookup[_nc(c)]:
                    return 'VCG (QAOA)'
            return 'exact'

        pc_all = all_layers_vc[all_layers_vc['method'] == 'PC-QAOA'].copy()
        pc_all['prep_type'] = pc_all.apply(_prep, axis=1)

        p_max_vc = int(all_layers_vc['layer'].max()) if 'layer' in all_layers_vc.columns else p_max

        lines.append("\n### VCG (QAOA-trained) vs exact Dicke-state prep — PC-QAOA only ###")
        for pt, grp in pc_all.groupby('prep_type'):
            n_inst = grp['constraints_hash'].nunique() if 'constraints_hash' in grp.columns else len(grp) // p_max_vc
            lines.append(f"  {pt}: {n_inst} instances ({len(grp)} experiment-layers)")

        for p_label, lyr in [('p=1', 1), (f'p={p_max_vc}', p_max_vc)]:
            src = _at_layer(pc_all, lyr)
            if src.empty:
                continue
            lines.append(f"\n  {p_label}:")
            for pt, grp in src.groupby('prep_type'):
                row = f"    {pt} (n={len(grp)}):"
                for metric in ['AR_feas', 'p_feasible', 'p_optimal']:
                    if metric in grp.columns:
                        row += f"  {metric}={grp[metric].mean():.3f}"
                lines.append(row)

        # ── VCG vs exact split by overlap_type ───────────────────────
        if 'overlap_type' in pc_all.columns:
            lines.append(f"\n  Split by overlap type:")
            for ot, grp_ot in pc_all.groupby('overlap_type'):
                lines.append(f"\n  [{ot}]")
                for pt, grp_pt in grp_ot.groupby('prep_type'):
                    n_inst = grp_pt['constraints_hash'].nunique() if 'constraints_hash' in grp_pt.columns else len(grp_pt) // p_max_vc
                    lines.append(f"    {pt}: {n_inst} instances")
                for p_label, lyr in [('p=1', 1), (f'p={p_max_vc}', p_max_vc)]:
                    src = _at_layer(grp_ot, lyr)
                    if src.empty:
                        continue
                    lines.append(f"    {p_label}:")
                    for pt, grp in src.groupby('prep_type'):
                        row = f"      {pt} (n={len(grp)}):"
                        for metric in ['AR_feas', 'p_feasible', 'p_optimal']:
                            if metric in grp.columns:
                                row += f"  {metric}={grp[metric].mean():.3f}"
                        lines.append(row)

    # ------------------------------------------------------------------
    # Circuit resources
    # ------------------------------------------------------------------
    if circ_df is not None and not circ_df.empty:
        circ = circ_df.copy()

        def _cstats(s):
            return f"mean={s.mean():.1f}  median={s.median():.1f}  min={s.min():.0f}  max={s.max():.0f}"

        lines.append("\n### Qubit counts ###")
        lines.append("  (PC-QAOA: no slack for structurally enforced constraints)")
        lines.append(f"  {'n':>3}  {'PC-QAOA qubits':>16}  {'slack':>6}  {'Penalty qubits':>16}  {'slack':>6}")
        for nx, grp in circ.groupby('n_x'):
            lines.append(f"  {nx:>3}  {grp['n_qubits_pc'].mean():>16.1f}  {grp['n_slack_pc'].mean():>6.1f}"
                         f"  {grp['n_qubits_p'].mean():>16.1f}  {grp['n_slack_p'].mean():>6.1f}")
        lines.append(f"\n  Overall:")
        lines.append(f"    PC-QAOA:     {_cstats(circ['n_qubits_pc'])}  (slack: {_cstats(circ['n_slack_pc'])})")
        lines.append(f"    PenaltyQAOA: {_cstats(circ['n_qubits_p'])}  (slack: {_cstats(circ['n_slack_p'])})")

        lines.append("\n### State-preparation gate counts ###")
        lines.append("  PC-QAOA = VCG/Dicke circuit; PenaltyQAOA = H^n product state.")
        lines.append(f"  {'n':>3}  {'PC total':>10}  {'PC 2q':>8}  {'Pen total':>10}  {'Pen 2q':>8}")
        for nx, grp in circ.groupby('n_x'):
            lines.append(f"  {nx:>3}  {grp['sp_total_pc'].mean():>10.1f}  {grp['sp_2q_pc'].mean():>8.1f}"
                         f"  {grp['sp_total_p'].mean():>10.1f}  {grp['sp_2q_p'].mean():>8.1f}")
        lines.append(f"\n  Overall:")
        lines.append(f"    PC-QAOA:     total {_cstats(circ['sp_total_pc'])}  2q {_cstats(circ['sp_2q_pc'])}")
        lines.append(f"    PenaltyQAOA: total {_cstats(circ['sp_total_p'])}  2q {_cstats(circ['sp_2q_p'])}")

        lines.append("\n### Per-QAOA-layer gate counts (×1 layer) ###")
        lines.append(f"  {'n':>3}  {'PC total':>10}  {'PC 2q':>8}  {'Pen total':>10}  {'Pen 2q':>8}")
        for nx, grp in circ.groupby('n_x'):
            lines.append(f"  {nx:>3}  {grp['layer_total_pc'].mean():>10.1f}  {grp['layer_2q_pc'].mean():>8.1f}"
                         f"  {grp['layer_total_p'].mean():>10.1f}  {grp['layer_2q_p'].mean():>8.1f}")
        lines.append(f"\n  Overall:")
        lines.append(f"    PC-QAOA:     total {_cstats(circ['layer_total_pc'])}  2q {_cstats(circ['layer_2q_pc'])}")
        lines.append(f"    PenaltyQAOA: total {_cstats(circ['layer_total_p'])}  2q {_cstats(circ['layer_2q_p'])}")

        p_vals = (1, 5)
        lines.append("\n### Total 2-qubit gates (state prep + p × layer) ###")
        for pv in p_vals:
            circ[f'_t2q_pc_{pv}'] = circ['sp_2q_pc'] + pv * circ['layer_2q_pc']
            circ[f'_t2q_p_{pv}']  = circ['sp_2q_p']  + pv * circ['layer_2q_p']
        lines.append(f"  {'n':>3}  " + "  ".join(
            f"{'PC p='+str(pv):>10}  {'Pen p='+str(pv):>10}" for pv in p_vals))
        for nx, grp in circ.groupby('n_x'):
            row = f"  {nx:>3}"
            for pv in p_vals:
                row += f"  {grp[f'_t2q_pc_{pv}'].mean():>10.1f}  {grp[f'_t2q_p_{pv}'].mean():>10.1f}"
            lines.append(row)
        lines.append(f"\n  Overall:")
        for pv in p_vals:
            ratio = circ[f'_t2q_p_{pv}'].mean() / circ[f'_t2q_pc_{pv}'].mean()
            lines.append(f"    p={pv}:  PC-QAOA {_cstats(circ[f'_t2q_pc_{pv}'])}  "
                         f"PenaltyQAOA {_cstats(circ[f'_t2q_p_{pv}'])}")
            lines.append(f"      PenaltyQAOA/PC-QAOA ratio (mean): {ratio:.2f}x")

        if 'has_vcg_pc' in circ.columns:
            lines.append("\n### PC-QAOA state-prep: VCG vs exact, by n ###")
            lines.append(f"  {'n':>3}  {'VCG instances':>14}  {'exact instances':>16}"
                         f"  {'VCG sp_2q mean':>15}  {'exact sp_2q mean':>17}")
            for nx, grp in circ.groupby('n_x'):
                vcg  = grp[grp['has_vcg_pc'] == True]
                exct = grp[grp['has_vcg_pc'] == False]
                lines.append(f"  {nx:>3}  {len(vcg):>14}  {len(exct):>16}"
                             f"  {vcg['sp_2q_pc'].mean() if len(vcg) else float('nan'):>15.1f}"
                             f"  {exct['sp_2q_pc'].mean() if len(exct) else float('nan'):>17.1f}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    text = "\n".join(lines) + "\n"
    with open(save_path, 'w') as f:
        f.write(text)
    print(text)
    print(f"  → saved to {save_path}")


def _run_comparison_plots(comp_ar: pd.DataFrame, comp_ar_raw: pd.DataFrame,
                          comp_res: pd.DataFrame, dirs: dict,
                          comp_ar_conv: pd.DataFrame = None,
                          comp_ar_all: pd.DataFrame = None,
                          exact_lookup: dict = None) -> None:
    """Generate all comparison AR and feasibility plots into *dirs*."""
    if comp_ar.empty:
        return

    # Use all-layers data for new parametric plots; fall back to final-layer data.
    df_all = comp_ar_all if (comp_ar_all is not None and not comp_ar_all.empty) else comp_ar

    print("\n--- Comparison AR plots ---")

    for method in comp_ar.get('method', pd.Series(dtype=str)).unique():
        sub = comp_ar[comp_ar['method'] == method]
        plot_ar.plot_ar_by_n(
            sub, title=f'{method}: AR vs n_x',
            save_path=os.path.join(dirs['ar'], f'{method.lower()}_ar_by_n.png'))

    if 'method' in comp_ar.columns and comp_ar['method'].nunique() > 1:
        plot_ar.plot_ar_comparison(
            comp_ar,
            save_path=os.path.join(dirs['ar'], 'pcqaoa_vs_penalty_ar.png'))
        if 'AR_feas' in comp_ar.columns:
            plot_ar.plot_ar_feas_comparison(
                comp_ar,
                save_path=os.path.join(dirs['ar'], 'pcqaoa_vs_penalty_ar_feas.png'))

    plot_ar.plot_ar_vs_layers(
        comp_ar,
        save_path=os.path.join(dirs['ar'], 'ar_vs_layers.png'))

    layers_src = comp_ar_conv if (comp_ar_conv is not None and not comp_ar_conv.empty) else comp_ar
    if 'n_layers' in layers_src.columns:
        plot_ar.plot_layers_to_threshold(
            layers_src,
            save_path=os.path.join(dirs['ar'], 'layers_to_threshold.png'))

    print("\n--- Parametric metric vs n_x plots ---")

    _METRICS = [
        ('AR_feas', r'Mean $\mathrm{AR}_{\mathrm{feas}}$',
         None, 'ar_feas', r'$\mathrm{AR}_{\mathrm{feas}}$ vs. Problem Size'),
        ('p_feasible', r'Mean $P(\mathrm{feas})$',
         None, 'p_feas', r'$P(\mathrm{feas})$ vs. Problem Size'),
        ('p_optimal', r'Mean $P(\mathrm{opt})$',
         None, 'p_opt', r'$P(\mathrm{opt})$ vs. Problem Size'),
    ]

    for metric, ylabel, threshold, stem, base_title in _METRICS:
        if metric not in df_all.columns:
            continue

        plot_feasibility.plot_metric_vs_nx(
            df_all, metric, ylabel, threshold=threshold,
            title=base_title,
            save_path=os.path.join(dirs['feasibility'], f'{stem}_vs_nx.png'))

        plot_feasibility.plot_metric_vs_nx_split(
            df_all, metric, ylabel, threshold=threshold,
            title=base_title + ' by Constraint Structure',
            save_path=os.path.join(dirs['feasibility'], f'{stem}_vs_nx_split.png'))

        plot_feasibility.plot_metric_vcg_vs_exact(
            df_all, metric, ylabel, threshold=threshold,
            title=base_title + r' (VCG vs. Exact)',
            exact_lookup=exact_lookup,
            save_path=os.path.join(dirs['feasibility'], f'{stem}_vcg_vs_exact.png'))

    if 'p_feasible' in comp_ar.columns and 'constraint_type' in comp_ar.columns:
        plot_feasibility.plot_p_feasible_by_family(
            comp_ar, df_all=df_all,
            save_path=os.path.join(dirs['feasibility'], 'p_feas_by_family.png'))

    if 'overlap_type' in comp_ar.columns:
        plot_feasibility.plot_p_feasible_with_overlap_split(
            comp_ar,
            save_path=os.path.join(dirs['feasibility'], 'pcqaoa_disjoint_vs_overlap.png'))

    if 'has_vcg' in comp_ar.columns:
        plot_feasibility.plot_vcg_exact_counts(
            comp_ar,
            save_path=os.path.join(dirs['feasibility'], 'vcg_exact_counts.png'))

    groupby = [c for c in ['method', 'constraint_type', 'n_x', 'layer'] if c in comp_ar.columns]
    metrics = [c for c in ['AR', 'AR_feas', 'p_feasible', 'p_optimal'] if c in comp_ar.columns]
    if groupby and metrics:
        summary_stats(comp_ar, groupby, metrics).to_csv(
            os.path.join(dirs['summaries'], 'comparison_ar_summary.csv'), index=False)


def _build_convergence_df(all_layers: pd.DataFrame, p_max: int) -> pd.DataFrame:
    """From all-layers data, build one row per experiment where n_layers = first
    layer where p_feasible >= 0.75, or p_max if never reached.

    This makes plot_layers_to_threshold and convergence paper-stats correct.
    """
    exp_keys = [c for c in ['method', 'constraints_hash', 'qubo_string', 'n_x', 'angle_strategy']
                if c in all_layers.columns]
    rows = []
    for _, grp in all_layers.groupby(exp_keys, sort=False):
        grp_s = grp.sort_values('layer')
        conv = grp_s[grp_s['p_feasible'] >= 0.75] if 'p_feasible' in grp_s.columns else pd.DataFrame()
        final = grp_s.iloc[-1].copy()
        final['n_layers'] = int(conv['layer'].min()) if len(conv) > 0 else p_max
        rows.append(final)
    return pd.DataFrame(rows).reset_index(drop=True)


def analyse(
    vcg_ar_path: str = None,
    vcg_res_path: str = None,
    vcg_db_path: str = None,
    vcg_circuit_res_path: str = None,
    comp_ar_path: str = None,
    comp_res_path: str = None,
    comp_ar_all_path: str = None,
    circ_csv_path: str = None,
    output_dir: str = './analysis_output/',
) -> None:
    """Generate all analysis plots and summary CSVs from pre-split result pickles.

    Expects pickles produced by ``split_results.py``.  Any path can be ``None``
    or missing — only the sections for which data is provided will run.

    For the PC-QAOA vs PenaltyQAOA comparison, ``comparison_ar.pkl`` is expected to
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
        Path to ``comparison_ar.pkl`` (PC-QAOA vs PenaltyQAOA AR data).
    comp_res_path : str, optional
        Path to ``comparison_resources.pkl`` (PC-QAOA vs PenaltyQAOA resource data).
    output_dir : str
        Root directory for all outputs (figures, summaries, stats).
        Subdirectories are created automatically via :func:`_makedirs`.
    """
    dirs = _makedirs(output_dir)

    vcg_ar = _load(vcg_ar_path, 'VCG AR')
    vcg_res = _load(vcg_res_path, 'VCG Resources')
    comp_ar = _load(comp_ar_path, 'Comparison AR')
    comp_res = _load(comp_res_path, 'Comparison Resources')
    comp_ar_all = _load(comp_ar_all_path, 'Comparison AR (all layers)')
    circ_df = pd.read_csv(circ_csv_path) if circ_csv_path and os.path.exists(circ_csv_path) else None

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
    # VCG gadget database plots
    # ------------------------------------------------------------------
    exact_lookup = None
    if vcg_db_path and os.path.exists(vcg_db_path):
        print("\n--- VCG gadget DB plots ---")
        vcg_df = load_db_as_df(vcg_db_path)
        # Build lookup for VCG-vs-exact classification in comparison plots
        import pickle as _pkl
        with open(vcg_db_path, 'rb') as _f:
            _raw_db = _pkl.load(_f)
        exact_lookup = {k: (e['opt_angles'] is None) for k, e in _raw_db.items()}
        if not comp_ar.empty:
            comp_ar = add_vcg_flag(comp_ar, exact_lookup)
        if not comp_ar_all.empty:
            comp_ar_all = add_vcg_flag(comp_ar_all, exact_lookup)
        print_summary(vcg_df)
        plot_ar_by_type(vcg_df, dirs['vcg_db'])
        plot_entropy_by_type(vcg_df, dirs['vcg_db'])
        plot_layers_by_type(vcg_df, dirs['vcg_db'])
        plot_layers_vs_nx(vcg_df, dirs['vcg_db'])
        plot_entropy_vs_layers(vcg_df, dirs['vcg_db'])
        plot_train_time(vcg_df, dirs['vcg_db'])
        plot_vcg_layers_to_entropy_threshold(vcg_df, dirs['vcg_db'])
        plot_convergence_scatter(vcg_df, dirs['vcg_db'])

        if vcg_circuit_res_path and os.path.exists(vcg_circuit_res_path):
            res_df = pd.read_pickle(vcg_circuit_res_path)
            plot_circuit_resources(res_df, dirs['vcg_db'])
        else:
            print(f'  [skip] vcg_circuit_resources.png — path not provided or missing')

        print("\n--- VCG paper stats ---")
        _generate_vcg_paper_stats(
            vcg_df,
            save_path=os.path.join(dirs['summaries'], 'vcg_paper_stats.txt'),
        )

    # ------------------------------------------------------------------
    # PC-QAOA vs PenaltyQAOA comparison analysis
    # ------------------------------------------------------------------
    comp_ar_raw = comp_ar.copy()
    n_total = len(comp_ar)
    if not comp_ar.empty:
        p_max = int(comp_ar['layer'].max()) if 'layer' in comp_ar.columns else int(comp_ar['n_layers'].max())
        # Build convergence-layer DataFrame from all-layers data if available;
        # otherwise fall back to the final-layer comp_ar (n_layers will be p_max for all).
        if not comp_ar_all.empty:
            comp_ar_conv = _build_convergence_df(comp_ar_all, p_max)
        else:
            comp_ar_conv = comp_ar.copy()

        completed = comp_ar[
            (comp_ar['p_feasible'] >= 0.75)
            | (comp_ar['layer'] == p_max)
        ].copy() if 'layer' in comp_ar.columns else comp_ar[
            (comp_ar['p_feasible'] >= 0.75)
            | (comp_ar['n_layers'] == p_max)
        ].copy()
        n_total = len(comp_ar)
        n_comp = len(completed)
        print(f"\n  Completed experiments: {n_comp}/{n_total} "
              f"({completed.groupby('method').size().to_dict()})")
        comp_ar = completed

        # All-instances plots (includes infeasible problems)
        _run_comparison_plots(comp_ar, comp_ar_raw, comp_res, dirs,
                              comp_ar_conv=comp_ar_conv,
                              comp_ar_all=comp_ar_all,
                              exact_lookup=exact_lookup)

        # Matched-pairs plots: only instances where both methods completed
        if 'constraints_hash' in comp_ar.columns:
            both_methods = (
                comp_ar.groupby('constraints_hash')['method'].nunique() == 2
            )
            matched_hashes = both_methods[both_methods].index
            comp_ar_matched = comp_ar[comp_ar['constraints_hash'].isin(matched_hashes)].copy()
            if not comp_ar_matched.empty:
                matched_dirs = _makedirs(os.path.join(output_dir, 'matched'))
                _run_comparison_plots(comp_ar_matched, comp_ar_raw, comp_res, matched_dirs,
                                      exact_lookup=exact_lookup)
                # Resource plot for matched
                if not comp_res.empty and 'constraints_hash' in comp_res.columns:
                    comp_res_m = comp_res[comp_res['constraints_hash'].isin(matched_hashes)].copy()
                    if not comp_res_m.empty:
                        plot_resources.plot_total_time_vs_nx(
                            comp_res_m, comp_ar_matched,
                            save_path=os.path.join(matched_dirs['resources'],
                                                   'comparison_total_time.png'))
                if 'has_feasible_solution' in comp_ar_matched.columns:
                    comp_ar_matched_feas = comp_ar_matched[comp_ar_matched['has_feasible_solution']].copy()
                    if not comp_ar_matched_feas.empty:
                        matched_feas_dirs = _makedirs(os.path.join(output_dir, 'matched_feasible_only'))
                        _run_comparison_plots(comp_ar_matched_feas, comp_ar_raw, comp_res,
                                              matched_feas_dirs, exact_lookup=exact_lookup)
                        if not comp_res.empty and 'constraints_hash' in comp_res.columns:
                            comp_res_mf = comp_res[comp_res['constraints_hash'].isin(
                                set(comp_ar_matched_feas['constraints_hash']))].copy()
                            if not comp_res_mf.empty:
                                plot_resources.plot_total_time_vs_nx(
                                    comp_res_mf, comp_ar_matched_feas,
                                    save_path=os.path.join(matched_feas_dirs['resources'],
                                                           'comparison_total_time.png'))

        # Feasible-instances-only plots (separate subdirectory)
        if 'has_feasible_solution' in comp_ar.columns:
            comp_ar_feas_only = comp_ar[comp_ar['has_feasible_solution']].copy()
            n_feas = len(comp_ar_feas_only)
            n_infeas = len(comp_ar) - n_feas
            print(f"\n  Feasible-only subset: {n_feas} rows "
                  f"({n_infeas} infeasible instances excluded)")
            if not comp_ar_feas_only.empty:
                feas_dirs = _makedirs(os.path.join(output_dir, 'feasible_only'))
                _run_comparison_plots(comp_ar_feas_only, comp_ar_raw, comp_res, feas_dirs,
                                      exact_lookup=exact_lookup)
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
    # Statistical tests (uses AR splits for both VCG and PC-QAOA)
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
        comp_ar_conv=comp_ar_conv,
        comp_ar_all=comp_ar_all,
        exact_lookup=exact_lookup,
        circ_df=circ_df,
    )

    print(f"\nAnalysis complete. Outputs in: {output_dir}")


def main() -> None:
    """CLI entry point.  Parse arguments and call :func:`analyse`."""
    parser = argparse.ArgumentParser(
        description='Analyse constraint_gadget experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--vcg-ar', default=None,
                        help='VCG AR split pickle (from split_results.py)')
    parser.add_argument('--vcg-res', default=None,
                        help='VCG resources split pickle')
    parser.add_argument('--vcg-db', default=None,
                        help='VCG gadget database pickle (vcg_db.pkl)')
    parser.add_argument('--vcg-circuit-res', default=None,
                        help='VCG circuit resources pickle (vcg_circuit_resources.pkl)')
    parser.add_argument('--comp-ar', default=None,
                        help='PC-QAOA vs PenaltyQAOA AR split pickle (final layer per experiment)')
    parser.add_argument('--comp-ar-all', default=None,
                        help='PC-QAOA vs PenaltyQAOA AR all-layers pickle (for convergence stats)')
    parser.add_argument('--comp-res', default=None,
                        help='PC-QAOA vs PenaltyQAOA resources split pickle')

    parser.add_argument('--output-dir', default='./analysis_output/',
                        help='Directory for all outputs (default: ./analysis_output/)')
    args = parser.parse_args()

    if not any([args.vcg_ar, args.vcg_res, args.vcg_db, args.comp_ar, args.comp_res]):
        parser.error('Provide at least one input file.')

    analyse(
        vcg_ar_path=args.vcg_ar,
        vcg_res_path=args.vcg_res,
        vcg_db_path=args.vcg_db,
        vcg_circuit_res_path=args.vcg_circuit_res,
        comp_ar_path=args.comp_ar,
        comp_res_path=args.comp_res,
        comp_ar_all_path=args.comp_ar_all,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
