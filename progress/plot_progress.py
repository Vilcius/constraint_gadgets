"""
plot_progress.py -- Visualise how P(opt) improves across algorithmic changes.

Figures produced
----------------
1. p_opt_vs_layers.png
   P(opt) vs QAOA depth p for Cases 1–4, comparing VCG(flag) [AR-only baseline]
   vs VCG(entropy).

2. h_norm_vs_depth.png
   H_norm vs training depth (number of ma-QAOA layers) for every non-trivial
   gadget, showing why entropy training sometimes needs more layers.

3. p_opt_summary_bar.png
   Grouped bar chart: best P(opt) across p=1..8 for each case & variant.

Run from project root:
    python progress/plot_progress.py
"""

import os, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_results import plot_utils as pu

os.makedirs('progress/figures', exist_ok=True)
pu.setup_style()

C = pu._ROSE_PINE

# ═══════════════════════════════════════════════════════════════════════════════
# Hardcoded results from completed runs
# ═══════════════════════════════════════════════════════════════════════════════

# ── VCG(flag) layer sweep  [AR-only training, run_focus_cases_output.txt] ──────
# p_opt[case_name][p] = P(opt)
VCG_FLAG = {
    'Case 1\nindep+knap+card': {
        1: 0.003, 2: 0.041, 3: 0.014, 4: 0.178, 5: 0.001,
        6: 0.007, 7: 0.023, 8: 0.002,
    },
    'Case 2\nknap+knap+card': {
        1: 0.003, 2: 0.009, 3: 0.013, 4: 0.017, 5: 0.000,
        6: 0.062, 7: 0.015, 8: 0.048,
    },
    'Case 3\ncard+knap': {
        1: 0.084, 2: 0.144, 3: 0.106, 4: 0.072, 5: 0.022,
        6: 0.148, 7: 0.162, 8: 0.317,
    },
    'Case 4\ncard+card': {
        1: 0.000, 2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000,
        6: 0.000, 7: 0.000, 8: 0.000,
    },
}

# ── VCG(entropy) layer sweep  [focus_run_entropy.log] ───────────────────
VCG_NOFLAG = {
    'Case 1\nindep+knap+card': {1: 0.410, 2: 0.889},
    'Case 2\nknap+knap+card':  {1: 0.534},
    'Case 3\ncard+knap':       {1: 0.216, 2: 0.605},
    'Case 4\ncard+card': {
        1: 0.000, 2: 0.000, 3: 0.000, 4: 0.000,
        5: 0.000, 6: 0.000, 7: 0.000, 8: 0.000,
    },
}

# ── Per-layer H_norm during gadget training  [plot_vcg_dist.log] ──────────────
# Only gadgets with |F| > 1 and multi-layer sweep are interesting here.
ENTROPY_TRAINING = {
    'x₀x₁=0\n(Case 1, |F|=3)': {
        1: 0.6592, 2: 0.0451, 3: 0.5880, 4: 0.8215, 5: 0.9245,
    },
    '2x₂+x₃+4x₄≤2\n(Case 1, |F|=3)': {
        1: 0.9963,
    },
    '3x₀+2x₁≤2\n(Case 2, |F|=2)': {
        1: 1.0000,
    },
    '2x₂+x₃≤2\n(Case 2, |F|=3)': {
        1: 0.7518, 2: 0.7208, 3: 0.8453, 4: 0.8484, 5: 0.7769, 6: 0.9223,
    },
    '5x₂+2x₃+5x₄+5x₅≤9\n(Case 3, |F|=8)': {
        1: 0.3420, 2: 0.3857, 3: 0.6915, 4: 0.7493,
        5: 0.6042, 6: 0.4865, 7: 0.8121, 8: 0.9386,
    },
    'x₃+x₄≥1\n(Case 4, |F|=3)': {
        1: 0.9108,
    },
}

# ── PenaltyQAOA best P(opt)  [server results] ─────────────────────────────────
PENALTY_BEST = {
    'Case 1\nindep+knap+card': 0.000,
    'Case 2\nknap+knap+card':  0.000,
    'Case 3\ncard+knap':       0.871,
    'Case 4\ncard+card':       0.000,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: P(opt) vs QAOA layers  (2×2 grid, one panel per case)
# ═══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

case_names = list(VCG_FLAG.keys())

for i, name in enumerate(case_names):
    ax = axes[i]

    flag_data  = VCG_FLAG[name]
    noflag_data = VCG_NOFLAG[name]
    pen_best    = PENALTY_BEST[name]

    flag_ps  = sorted(flag_data)
    flag_vals = [flag_data[p] for p in flag_ps]

    noflag_ps   = sorted(noflag_data)
    noflag_vals = [noflag_data[p] for p in noflag_ps]

    ax.plot(flag_ps, flag_vals,
            color=C['pine'], marker='o', linewidth=1.8, markersize=5,
            label='VCG(flag) — AR-only')
    ax.plot(noflag_ps, noflag_vals,
            color=C['iris'], marker='s', linewidth=1.8, markersize=5,
            label='VCG — entropy')
    if pen_best > 0:
        ax.axhline(pen_best, color=C['love'], linestyle='--', linewidth=1.4,
                   label=f'PenaltyQAOA best ({pen_best:.3f})')

    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('QAOA depth  $p$', fontsize=10)
    ax.set_ylabel('$P$(opt)', fontsize=10)
    short = name.split('\n')[0]
    ax.set_title(short, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

fig1.suptitle('$P$(opt) vs QAOA depth: VCG(flag) vs VCG(entropy)',
              fontsize=12, y=1.01)
fig1.tight_layout()
out1 = 'progress/figures/p_opt_vs_layers.png'
fig1.savefig(out1, bbox_inches='tight', dpi=150)
plt.close(fig1)
print(f'Saved → {out1}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: H_norm vs training depth per gadget
# ═══════════════════════════════════════════════════════════════════════════════

# Only show gadgets with more than 1 training layer
multi_layer = {k: v for k, v in ENTROPY_TRAINING.items() if len(v) > 1}

n = len(multi_layer)
ncols = 2
nrows = (n + 1) // 2

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows))
axes2 = np.array(axes2).flatten()

for i, (gadget_name, h_per_layer) in enumerate(multi_layer.items()):
    ax = axes2[i]
    layers = sorted(h_per_layer)
    vals   = [h_per_layer[p] for p in layers]

    ax.plot(layers, vals, color=C['iris'], marker='o',
            linewidth=1.8, markersize=6, zorder=3)
    # Mark the selected best layer
    best_p = max(h_per_layer, key=h_per_layer.get)
    ax.scatter([best_p], [h_per_layer[best_p]],
               color=C['gold'], s=80, zorder=4, label=f'Selected $p={best_p}$')

    ax.axhline(0.9, color=C['muted'], linestyle=':', linewidth=1.2,
               label='Threshold (0.90)')
    ax.axhline(1.0, color=C['foam'],  linestyle='--', linewidth=1.0, alpha=0.6)

    ax.set_xlim(0.5, max(layers) + 0.5)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('Training depth  $p$', fontsize=9)
    ax.set_ylabel('$H_\\mathrm{norm}$', fontsize=9)
    ax.set_title(gadget_name, fontsize=8.5)
    ax.legend(fontsize=8)

# Hide unused panels
for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle('$H_\\mathrm{norm}$ vs training depth for non-trivial VCG gadgets\n'
              '(gold dot = selected depth; dotted line = entropy threshold)',
              fontsize=11, y=1.01)
fig2.tight_layout()
out2 = 'progress/figures/h_norm_vs_depth.png'
fig2.savefig(out2, bbox_inches='tight', dpi=150)
plt.close(fig2)
print(f'Saved → {out2}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Summary grouped bar chart — best P(opt) per case per variant
# ═══════════════════════════════════════════════════════════════════════════════

short_labels = ['Case 1', 'Case 2', 'Case 3', 'Case 4']

# Best P(opt) across all layers for each variant
flag_best   = [max(VCG_FLAG[n].values())   for n in case_names]
vcg_best = [max(VCG_NOFLAG[n].values()) for n in case_names]
pen_best_v  = [PENALTY_BEST[n]              for n in case_names]

x = np.arange(len(short_labels))
w = 0.25

fig3, ax3 = plt.subplots(figsize=(9, 5))
bars1 = ax3.bar(x - w,     pen_best_v,  w, label='PenaltyQAOA',          color=C['love'],  alpha=0.85)
bars2 = ax3.bar(x,         flag_best,   w, label='VCG(flag)  AR-only',    color=C['pine'],  alpha=0.85)
bars3 = ax3.bar(x + w,     vcg_best, w, label='VCG  entropy',    color=C['iris'],  alpha=0.85)

# Value labels on bars
for bars in (bars1, bars2, bars3):
    for bar in bars:
        h = bar.get_height()
        if h > 0.005:
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

ax3.set_xticks(x)
ax3.set_xticklabels(short_labels, fontsize=11)
ax3.set_ylim(0, 1.15)
ax3.set_ylabel('Best $P$(opt)  over $p = 1 \\ldots 8$', fontsize=10)
ax3.set_title('Best $P$(opt) per method — Cases 1–4\n'
              '(PenaltyQAOA uses best-p from server results)',
              fontsize=11)
ax3.legend(fontsize=9, loc='upper right')
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

out3 = 'progress/figures/p_opt_summary_bar.png'
fig3.tight_layout()
fig3.savefig(out3, bbox_inches='tight', dpi=150)
plt.close(fig3)
print(f'Saved → {out3}')
