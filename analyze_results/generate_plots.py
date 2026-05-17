"""
generate_plots.py — combine disjoint + overlapping results, run analysis, copy
plots to paper/figures/plots/.

Usage (from repo root, in qaoa_jax env):
    python analyze_results/generate_plots.py
"""

import os
import shutil
import pickle
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analyze_results.main_analysis import (analyse, _generate_paper_stats,
                                            _build_convergence_df)
from analyze_results import plot_resources
from analyze_results.generate_results_markdown import generate as generate_markdown

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIS = os.path.join(ROOT, 'results', 'disjoint')
RESULTS_OV = os.path.join(ROOT, 'results', 'overlapping')
COMBINED = os.path.join(ROOT, 'results', 'combined')
OUTPUT_DIR = os.path.join(ROOT, 'analysis_output', 'combined')
VCG_DB = os.path.join(ROOT, 'gadgets', 'vcg_db.pkl')
VCG_RES = os.path.join(ROOT, 'results', 'vcg_circuit_resources.pkl')
PAPER_PLOTS = os.path.join(ROOT, '..', 'paper', 'figures', 'plots')
PAPER_NUMERICS = os.path.join(ROOT, '..', 'paper', 'numerics')

os.makedirs(COMBINED, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def combine(filename, label_dis='disjoint', label_ov='overlapping'):
    """Load filename from both sets, tag with overlap_type, concatenate."""
    dis = load(os.path.join(RESULTS_DIS, filename))
    ov = load(os.path.join(RESULTS_OV, filename))
    dis['overlap_type'] = label_dis
    ov['overlap_type'] = label_ov
    return pd.concat([dis, ov], ignore_index=True)


def save(df, path):
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    print(f'  saved {path}  ({len(df)} rows)')


# ── combine ───────────────────────────────────────────────────────────────────
print('=== Combining results ===')

comp_ar_path = os.path.join(COMBINED, 'comparison_ar.pkl')
comp_ar_all_path = os.path.join(COMBINED, 'comparison_ar_all_layers.pkl')
comp_res_path = os.path.join(COMBINED, 'comparison_resources.pkl')

save(combine('comparison_ar.pkl'), comp_ar_path)
save(combine('comparison_ar_all_layers.pkl'), comp_ar_all_path)
save(combine('comparison_resources.pkl'), comp_res_path)

circ_dis = pd.read_csv(os.path.join(RESULTS_DIS, 'circuit_resources.csv'))
circ_ov  = pd.read_csv(os.path.join(RESULTS_OV,  'circuit_resources.csv'))
circ_combined = pd.concat([circ_dis, circ_ov], ignore_index=True)
circ_combined_path = os.path.join(COMBINED, 'circuit_resources.csv')
circ_combined.to_csv(circ_combined_path, index=False)
print(f'  saved {circ_combined_path}  ({len(circ_combined)} rows)')

# ── run analysis ──────────────────────────────────────────────────────────────
print('\n=== Running analysis ===')

analyse(
    comp_ar_path=comp_ar_path,
    comp_ar_all_path=comp_ar_all_path,
    comp_res_path=comp_res_path,
    vcg_db_path=VCG_DB,
    vcg_circuit_res_path=VCG_RES,
    circ_csv_path=circ_combined_path,
    output_dir=OUTPUT_DIR,
)

# Circuit resources figure (combines both sets)
print('\n=== Circuit resources plot ===')
circ_out = os.path.join(OUTPUT_DIR, 'figures', 'resources', 'circuit_resources_vs_nx.png')
plot_resources.plot_circuit_resources_vs_nx(circ_combined, save_path=circ_out)
print(f'  saved circuit_resources_vs_nx.png')

# ── copy plots to paper ───────────────────────────────────────────────────────
print(f'\n=== Copying plots to {PAPER_PLOTS} ===')

os.makedirs(PAPER_PLOTS, exist_ok=True)

AR_DIR = os.path.join(OUTPUT_DIR, 'figures', 'ar')
FEAS_DIR = os.path.join(OUTPUT_DIR, 'figures', 'feasibility')
RES_DIR = os.path.join(OUTPUT_DIR, 'figures', 'resources')

PAPER_FIGURES = [
    # AR
    (AR_DIR, 'layers_to_threshold.png'),
    # Feasibility / metric vs nx (new parametric plots)
    (FEAS_DIR, 'ar_feas_vs_nx.png'),
    (FEAS_DIR, 'ar_feas_vs_nx_split.png'),
    (FEAS_DIR, 'ar_feas_vcg_vs_exact.png'),
    (FEAS_DIR, 'p_feas_vs_nx.png'),
    (FEAS_DIR, 'p_feas_vs_nx_split.png'),
    (FEAS_DIR, 'p_feas_vcg_vs_exact.png'),
    (FEAS_DIR, 'p_opt_vs_nx.png'),
    (FEAS_DIR, 'p_opt_vs_nx_split.png'),
    (FEAS_DIR, 'p_opt_vcg_vs_exact.png'),
    (FEAS_DIR, 'p_feas_by_family.png'),
    (FEAS_DIR, 'pcqaoa_disjoint_vs_overlap.png'),
    (FEAS_DIR, 'vcg_exact_counts.png'),
    # Resources
    (RES_DIR, 'circuit_resources_vs_nx.png'),
    (RES_DIR, 'comparison_shots_vs_n.png'),
    (RES_DIR, 'comparison_time_breakdown.png'),
    # VCG gadget plots
    (os.path.join(OUTPUT_DIR, 'figures', 'vcg_db'), 'vcg_entropy_by_type.png'),
    (os.path.join(OUTPUT_DIR, 'figures', 'vcg_db'), 'vcg_circuit_resources.png'),
    (os.path.join(OUTPUT_DIR, 'figures', 'vcg_db'), 'vcg_convergence_scatter.png'),
]

for src_dir, fname in PAPER_FIGURES:
    src = os.path.join(src_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(PAPER_PLOTS, fname))
        print(f'  {fname}')
    else:
        print(f'  MISSING: {fname}')

print(f'\nDone. Plots in {os.path.abspath(PAPER_PLOTS)}')

# ── per-split paper stats (disjoint / overlapping) ───────────────────────────
print('\n=== Per-split paper stats ===')

import pandas as _pd

for split_label, res_dir, out_dir, _circ_split in [
    ('disjoint',    RESULTS_DIS, os.path.join(ROOT, 'analysis_output', 'disjoint'),    circ_dis),
    ('overlapping', RESULTS_OV,  os.path.join(ROOT, 'analysis_output', 'overlapping'), circ_ov),
]:
    _ar_raw  = _pd.read_pickle(os.path.join(res_dir, 'comparison_ar.pkl'))
    _ar_all  = _pd.read_pickle(os.path.join(res_dir, 'comparison_ar_all_layers.pkl'))
    _res     = _pd.read_pickle(os.path.join(res_dir, 'comparison_resources.pkl'))

    _p_max = int(_ar_all['layer'].max()) if 'layer' in _ar_all.columns else int(_ar_all['n_layers'].max())
    _ar_conv = _build_convergence_df(_ar_all, _p_max)

    # Keep only experiments that completed (P(feas)>=0.75 or exhausted p_max)
    if 'layer' in _ar_raw.columns:
        _ar = _ar_raw[(_ar_raw['p_feasible'] >= 0.75) | (_ar_raw['layer'] == _p_max)].copy()
    else:
        _ar = _ar_raw[(_ar_raw['p_feasible'] >= 0.75) | (_ar_raw['n_layers'] == _p_max)].copy()

    os.makedirs(os.path.join(out_dir, 'summaries'), exist_ok=True)
    _save_path = os.path.join(out_dir, 'summaries', 'paper_stats.txt')

    _generate_paper_stats(
        comp_ar_raw=_ar_raw,
        comp_ar=_ar,
        comp_res=_res,
        vcg_ar=_pd.DataFrame(),
        n_total_raw=len(_ar_raw),
        save_path=_save_path,
        comp_ar_conv=_ar_conv,
        comp_ar_all=_ar_all,
        circ_df=_circ_split,
    )
    print(f'  [{split_label}] paper_stats.txt')

# ── copy stats to paper/numerics ──────────────────────────────────────────────
SUMMARIES_DIR = os.path.join(OUTPUT_DIR, 'summaries')

STATS_SOURCES = [
    (SUMMARIES_DIR,                                                         'paper_stats.txt',     'paper_stats.txt'),
    (SUMMARIES_DIR,                                                         'vcg_paper_stats.txt', 'vcg_paper_stats.txt'),
    (os.path.join(ROOT, 'analysis_output', 'disjoint',    'summaries'),    'paper_stats.txt',     'paper_stats_disjoint.txt'),
    (os.path.join(ROOT, 'analysis_output', 'overlapping', 'summaries'),    'paper_stats.txt',     'paper_stats_overlapping.txt'),
]

# Central collection in code dir
CODE_NUMERICS = os.path.join(ROOT, 'analysis_output', 'summaries')
os.makedirs(CODE_NUMERICS, exist_ok=True)
print(f'\n=== Gathering stats in {CODE_NUMERICS} ===')
for src_dir, fname, dest_fname in STATS_SOURCES:
    src = os.path.join(src_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(CODE_NUMERICS, dest_fname))
        print(f'  {dest_fname}')
    else:
        print(f'  MISSING: {dest_fname}')

# Copy to paper/numerics
print(f'\n=== Copying stats to {PAPER_NUMERICS} ===')
os.makedirs(PAPER_NUMERICS, exist_ok=True)
for src_dir, fname, dest_fname in STATS_SOURCES:
    src = os.path.join(src_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(PAPER_NUMERICS, dest_fname))
        print(f'  {dest_fname}')
    else:
        print(f'  MISSING: {dest_fname}')

print(f'Done. Stats in {os.path.abspath(CODE_NUMERICS)} and {os.path.abspath(PAPER_NUMERICS)}')

# ── generate GitHub results markdown ─────────────────────────────────────────
print('\n=== Generating results markdown ===')
for split in ('disjoint', 'overlapping'):
    generate_markdown(split)
print('Done.')
