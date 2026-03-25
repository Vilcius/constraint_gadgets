"""
plot_vcg_distributions.py -- VCG output distributions with feasible-state markers.

For every non-trivial gadget (|F| > 1) trained with entropy maximisation this script:
  1. Trains a VCG gadget.
  2. Samples 50k shots from opt_circuit().
  3. Plots the probability distribution as a bar chart:
       green  = feasible state
       red    = infeasible state
     with a dashed line at the uniform-over-feasible level and
     vertical dashed gold lines at every feasible state.
  4. Annotates each panel with AR, H_norm, |F|, n_qubits, depth p.

Output: progress/figures/vcg_distributions.png

Run from project root:
    python progress/plot_vcg_distributions.py
"""

import sys, os, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pennylane as qml
from pennylane import numpy as np

from core import constraint_handler as ch
from core.vcg import VCG
from analyze_results import plot_utils as pu

os.makedirs('progress/figures', exist_ok=True)

# ── Training budget ────────────────────────────────────────────────────────────
SHOTS            = 50_000
NF_AR_THRESHOLD  = 0.999
NF_ENT_THRESHOLD = 0.9
NF_MAX_LAYERS    = 8
NF_QAOA_RESTARTS = 10
NF_QAOA_STEPS    = 200
NF_MA_RESTARTS   = 20
NF_MA_STEPS      = 150
NF_LR            = 0.05

# ── Focus-case constraints (same as run_focus_cases.py) ───────────────────────
FOCUS_CASES = [
    {
        'label': 'Case 1',
        'constraints': ['x_0*x_1 == 0', '2*x_2 + 1*x_3 + 4*x_4 <= 2', 'x_5 + x_6 == 2'],
    },
    {
        'label': 'Case 2',
        'constraints': ['3*x_0 + 2*x_1 <= 2', '2*x_2 + 1*x_3 <= 2', 'x_4 + x_5 + x_6 >= 3'],
    },
    {
        'label': 'Case 3',
        'constraints': ['x_0 + x_1 >= 2', '5*x_2 + 2*x_3 + 5*x_4 + 5*x_5 <= 9'],
    },
    {
        'label': 'Case 4',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_3 + x_4 >= 1'],
    },
    {
        'label': 'Case 5',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_0 + x_3 + x_4 == 1', 'x_5 + x_6 >= 1'],
    },
    {
        'label': 'Case 6',
        'constraints': ['3*x_0 + 2*x_1 <= 2', 'x_1 + x_2 + x_3 <= 2', 'x_4 + x_5 + x_6 == 2'],
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_vcg_eligible(pc):
    return (
        not ch.is_dicke_compatible(pc)
        and not ch.is_cardinality_leq_compatible(pc)
        and not ch.is_flow_compatible(pc)
    )


# ── Collect non-trivial gadget constraints ────────────────────────────────────
gadgets = []   # (case_label, raw_str, VCG)

for task in FOCUS_CASES:
    parsed = ch.parse_constraints(task['constraints'])
    str_idx, _ = ch.partition_constraints(parsed, strategy='auto')
    for i in str_idx:
        if is_vcg_eligible(parsed[i]):
            raw = parsed[i].raw
            print(f'[{task["label"]}]  Training: {raw}', flush=True)
            vcg = VCG(
                constraints=[raw],
                ar_threshold=NF_AR_THRESHOLD,
                entropy_threshold=NF_ENT_THRESHOLD,
                max_layers=NF_MAX_LAYERS,
                qaoa_restarts=NF_QAOA_RESTARTS,
                qaoa_steps=NF_QAOA_STEPS,
                ma_restarts=NF_MA_RESTARTS,
                ma_steps=NF_MA_STEPS,
                lr=NF_LR,
                samples=SHOTS,
            )
            vcg.train(verbose=True)
            n_feas = vcg.n_feasible
            ent_str = f'{vcg.entropy:.4f}' if vcg.entropy is not None else 'N/A'
            print(f'  → AR={vcg.ar:.4f}  H_norm={ent_str}  '
                  f'layers={vcg.n_layers}  n_feas={n_feas}', flush=True)
            # Skip trivially single-state gadgets — they're just X gates
            if n_feas > 1:
                gadgets.append((task['label'], raw, vcg))

if not gadgets:
    print('No non-trivial VCG gadgets found.')
    sys.exit(0)

# ── Plot ───────────────────────────────────────────────────────────────────────
C = pu._ROSE_PINE
COL_FEAS   = C['pine']      # blue-ish green for feasible
COL_INFEAS = C['love']      # red for infeasible
COL_UNIF   = C['muted']     # dashed uniform line
COL_VLINE  = C['gold']      # vertical markers at feasible states

pu.setup_style()

n_panels = len(gadgets)
ncols    = min(n_panels, 2)
nrows    = math.ceil(n_panels / ncols)
fig_w    = 7.0 * ncols
fig_h    = 3.8 * nrows + 0.5

fig = plt.figure(figsize=(fig_w, fig_h))
gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.70, wspace=0.40)

for idx, (case_label, raw, vcg) in enumerate(gadgets):
    row, col = divmod(idx, ncols)
    ax = fig.add_subplot(gs[row, col])

    # ── Sample distribution ────────────────────────────────────────────────
    counts = vcg.do_counts_circuit(shots=SHOTS)
    total  = sum(counts.values())
    n_x    = vcg.n_x

    all_states = [format(i, f'0{n_x}b') for i in range(2 ** n_x)]
    probs  = [counts.get(s, 0) / total for s in all_states]
    colors = [COL_FEAS if vcg._is_feasible(s) else COL_INFEAS for s in all_states]

    # Uniform-over-feasible reference
    n_feas = vcg.n_feasible
    p_unif = 1.0 / n_feas if n_feas > 0 else 0.0

    xs = range(len(all_states))
    ax.bar(xs, probs, color=colors, width=0.8, linewidth=0, zorder=2)

    # Vertical dashed lines at each feasible state
    for i, s in enumerate(all_states):
        if vcg._is_feasible(s):
            ax.axvline(i, color=COL_VLINE, linestyle='--', linewidth=1.2,
                       alpha=0.85, zorder=3)

    ax.axhline(p_unif, color=COL_UNIF, linestyle=':', linewidth=1.6, zorder=4,
               label=f'Uniform  (1/{n_feas})')

    # ── Axis formatting ────────────────────────────────────────────────────
    ax.set_xticks(range(len(all_states)))
    ax.set_xticklabels(all_states, rotation=90, fontsize=6,
                       fontfamily='monospace')
    ax.set_xlim(-0.5, len(all_states) - 0.5)
    ax.set_ylabel('Probability', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)

    ent_val = vcg.entropy
    ent_str = f'$H_\\mathrm{{norm}}={ent_val:.3f}$' if ent_val is not None else ''
    short_raw = raw if len(raw) <= 32 else raw[:30] + '…'
    title = (f'{case_label}  |  {short_raw}\n'
             f'AR={vcg.ar:.4f}   {ent_str}   '
             f'$|F|={n_feas}$   $n={n_x}$   $p={vcg.n_layers}$')
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.legend(fontsize=7.5, handlelength=1.6, loc='upper right')

    # Light shading across the feasible region (spanning min→max feasible index)
    feas_idx = [i for i, s in enumerate(all_states) if vcg._is_feasible(s)]
    if feas_idx:
        ax.axvspan(min(feas_idx) - 0.5, max(feas_idx) + 0.5,
                   color=COL_FEAS, alpha=0.05, zorder=1)

# ── Shared legend ──────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=COL_FEAS,   label='Feasible state'),
    mpatches.Patch(color=COL_INFEAS, label='Infeasible state'),
    plt.Line2D([0], [0], color=COL_VLINE, linestyle='--', linewidth=1.4,
               label='Feasible state positions'),
    plt.Line2D([0], [0], color=COL_UNIF, linestyle=':', linewidth=1.6,
               label='Uniform over feasible'),
]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    'VCG output distributions (entropy-maximising training)\n'
    'Gold dashed lines mark feasible states; dotted line = ideal uniform reference',
    fontsize=10, y=1.01,
)

out = 'progress/figures/vcg_distributions.png'
pu.save_fig(fig, out)
print(f'\nSaved → {out}', flush=True)
