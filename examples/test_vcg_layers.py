"""
test_vcg_layers.py -- VCG layer sweep: QAOA warm-start then ma-QAOA sweep.

Training procedure:
  1. Single QAOA run at p=1 (2 parameters, fast) to obtain warm-start angles.
  2. ma-QAOA layer sweep:
       p=1 : first restart seeded from QAOA p=1 angles (broadcast γ/β).
             Subsequent restarts random.
       p>1 : joint re-optimisation of all p*(num_gamma+num_beta) parameters,
             warm-started from the previous depth's optimal angles.
     Stops when AR >= THRESHOLD or MAX_LAYERS is reached.

Constraint types compared (both 5 variables, 1 flag qubit):
  - Linear knapsack    (5 vars): compact Pauli structure due to dominance
  - Quadratic knapsack (5 vars): dense Pauli structure from cross terms

Saves:
  examples/results/vcg_layer_sweep.pkl               -- DataFrame (all runs)
  examples/figures/vcg_layer_sweep_ar.png            -- AR vs depth
  examples/figures/vcg_layer_sweep_time.png          -- Optimisation time vs depth
  examples/figures/vcg_layer_sweep_distributions.png -- Measurement distributions

Run from project root:
    python examples/test_vcg_layers.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pennylane import numpy as np

from core import vcg as vcg_module
from analyze_results.results_helper import (
    ResultsCollector, read_typed_csv, collect_vcg_data,
)
from analyze_results import plot_utils as pu

os.makedirs('examples/results', exist_ok=True)
os.makedirs('examples/figures', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ══════════════════════════════════════════════════════════════════════════════

knap_rows  = read_typed_csv('data/knapsack_constraints.csv')
qknap_rows = read_typed_csv('data/quadratic_knapsack_constraints.csv')

CONSTRAINTS = {
    'knapsack':      next(cs[0] for n, cs in knap_rows  if n == 5),
    'quad_knapsack': next(cs[0] for n, cs in qknap_rows if n == 5),
}

THRESHOLD  = 0.95   # AR target
MAX_LAYERS = 8      # give up after this many layers
LR         = 0.05   # Adam learning rate
SHOTS      = 10_000 # measurement shots for distributions
FLAG_WIRE  = 5      # flag qubit index (one per 5-var constraint)

# QAOA p=1 warm-up budget (fast)
QAOA_RESTARTS = 5
QAOA_STEPS    = 150

# ma-QAOA budget per layer
MA_RESTARTS = 20
MA_STEPS    = 200

RESULTS_PATH = 'examples/results/vcg_layer_sweep.pkl'


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sweep
# ══════════════════════════════════════════════════════════════════════════════

collector = ResultsCollector()

for ctype, constraint in CONSTRAINTS.items():
    # Inspect Hamiltonian structure once (cheap: 1 step, 1 restart)
    probe = vcg_module.VCG(
        constraints=[constraint], flag_wires=[FLAG_WIRE],
        angle_strategy='ma-QAOA', decompose=True, n_layers=1, steps=1, num_restarts=1,
    )
    n_good  = probe.outcomes.count(-1)
    n_bad   = probe.outcomes.count(1)
    n_pauli = probe.num_gamma

    print('=' * 70, flush=True)
    print(f'Constraint type : {ctype}', flush=True)
    print(f'Constraint      : {constraint[:80]}', flush=True)
    print(f'States          : {len(probe.outcomes)} total,  {n_good} good,  {n_bad} bad',
          flush=True)
    print(f'Pauli terms     : {n_pauli}   (wires: {probe.num_beta})', flush=True)
    print('=' * 70, flush=True)

    # ── Step 1: single QAOA p=1 run ──────────────────────────────────────────
    print(f'\n  QAOA p=1  (restarts={QAOA_RESTARTS}, steps={QAOA_STEPS})', flush=True)
    qaoa_gadget = vcg_module.VCG(
        constraints=[constraint], flag_wires=[FLAG_WIRE],
        angle_strategy='QAOA', decompose=False,
        n_layers=1, steps=QAOA_STEPS, num_restarts=QAOA_RESTARTS, learning_rate=LR,
    )
    qaoa_cost, _ = qaoa_gadget.optimize_angles(qaoa_gadget.do_evolution_circuit)
    qaoa_ar = (float(qaoa_cost) - 1.0) / -2.0
    qaoa_angles = qaoa_gadget.opt_angles  # shape (1, 2)
    print(f'    AR={qaoa_ar:.4f}  time={qaoa_gadget.optimize_time:.1f}s', flush=True)

    # Store the QAOA p=1 row for comparison in plots
    qaoa_row = collect_vcg_data(
        qaoa_gadget, constraint_type=ctype, skip_optimize=True, shots=SHOTS,
    )
    qaoa_row['threshold_reached'] = [qaoa_ar >= THRESHOLD]
    qaoa_row['n_params']          = [2]
    collector.add(qaoa_row)

    # ── Step 2: ma-QAOA layer sweep ───────────────────────────────────────────
    print(f'  ma-QAOA sweep  (restarts={MA_RESTARTS}, steps={MA_STEPS})', flush=True)
    reached = False
    prev_best_ma = None

    for n_layers in range(1, MAX_LAYERS + 1):
        gadget = vcg_module.VCG(
            constraints=[constraint], flag_wires=[FLAG_WIRE],
            angle_strategy='ma-QAOA', decompose=True,
            n_layers=n_layers, steps=MA_STEPS, num_restarts=MA_RESTARTS, learning_rate=LR,
        )
        n_params = n_layers * (gadget.num_gamma + gadget.num_beta)

        if prev_best_ma is None:
            # p=1: seed first restart from QAOA p=1 angles
            opt_cost, _ = gadget.optimize_angles(
                gadget.do_evolution_circuit,
                starting_angles_from_qaoa=qaoa_angles,
            )
        else:
            # p>1: joint re-opt, warm-start from previous depth
            opt_cost, _ = gadget.optimize_angles(
                gadget.do_evolution_circuit,
                prev_layer_angles=prev_best_ma,
            )

        prev_best_ma = gadget.opt_angles
        ar = (float(opt_cost) - 1.0) / -2.0

        status = '✓' if ar >= THRESHOLD else ' '
        print(
            f'  {status} p={n_layers}: AR={ar:.4f}  params={n_params:3d}'
            f'  time={gadget.optimize_time:.1f}s',
            flush=True,
        )

        row = collect_vcg_data(
            gadget, constraint_type=ctype, skip_optimize=True, shots=SHOTS,
        )
        row['threshold_reached'] = [ar >= THRESHOLD]
        row['n_params']          = [n_params]
        collector.add(row)

        if ar >= THRESHOLD:
            reached = True
            break

    if not reached:
        print(f'    Did not reach AR >= {THRESHOLD} within {MAX_LAYERS} layers.',
              flush=True)

    print(flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Save
# ══════════════════════════════════════════════════════════════════════════════

collector.save(RESULTS_PATH)
df = collector.to_dataframe()

def _unwrap(df):
    """Unwrap single-element list cells left by collect_vcg_data."""
    for col in df.columns:
        if df.empty:
            break
        sample = df[col].iloc[0]
        if isinstance(sample, list) and len(sample) == 1:
            df[col] = df[col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
            )
    return df

df = _unwrap(df)
print(f'Saved {len(df)} rows to {RESULTS_PATH}\n', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Summary table
# ══════════════════════════════════════════════════════════════════════════════

print('Summary (all runs)')
print('-' * 80)
print(f"  {'Constraint':<16} {'Strategy':<10} {'p':>2}  {'AR':>7}  "
      f"{'params':>6}  {'Pauli terms':>11}  {'time(s)':>8}  {'thresh':>6}")
print('  ' + '-' * 70)
for _, r in df.sort_values(['constraint_type', 'angle_strategy', 'n_layers']).iterrows():
    mark = '✓' if r['threshold_reached'] else ' '
    print(
        f"  {mark} {r['constraint_type']:<16} {r['angle_strategy']:<10}"
        f" {r['n_layers']:>2}  {r['AR']:>7.4f}  {r['n_params']:>6}"
        f"  {r['num_gamma']:>11}  {r['optimize_time']:>8.1f}  {str(r['threshold_reached']):>6}",
    )
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Plots
# ══════════════════════════════════════════════════════════════════════════════

CTYPES = list(CONSTRAINTS.keys())


def plot_ar_sweep(df, save_path: str) -> None:
    """ma-QAOA AR vs depth, with QAOA p=1 baseline, one panel per constraint."""
    pu.setup_style()
    fig, axes = plt.subplots(1, len(CTYPES), figsize=(6 * len(CTYPES), 5), sharey=True)

    for ax, ctype in zip(axes, CTYPES):
        sub = df[df['constraint_type'] == ctype]

        # QAOA p=1 reference (horizontal dashed line)
        qaoa_row = sub[sub['angle_strategy'] == 'QAOA']
        if not qaoa_row.empty:
            qaoa_ar = qaoa_row.iloc[0]['AR']
            ax.axhline(qaoa_ar, color=pu.ANGLE_COLORS['QAOA'],
                       linestyle='--', linewidth=1.5, label=f'QAOA p=1 ({qaoa_ar:.3f})')

        # ma-QAOA layer sweep
        ma_grp = sub[sub['angle_strategy'] == 'ma-QAOA'].sort_values('n_layers')
        if not ma_grp.empty:
            ax.plot(ma_grp['n_layers'], ma_grp['AR'],
                    marker='o', color=pu.ANGLE_COLORS['ma-QAOA'],
                    label='ma-QAOA', linewidth=2)

        ax.axhline(THRESHOLD, color=pu._ROSE_PINE['muted'],
                   linestyle=':', linewidth=1.2, label=f'Threshold {THRESHOLD}')
        ax.set_title(ctype, fontsize=11)
        ax.set_xlabel('ma-QAOA layers (p)')
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Approximation Ratio (AR)')
    fig.suptitle('VCG: ma-QAOA AR vs depth  (QAOA p=1 warm-start)')
    pu.save_fig(fig, save_path)
    print(f'Saved: {save_path}')


def plot_time_sweep(df, save_path: str) -> None:
    """ma-QAOA optimisation time vs depth, one panel per constraint type."""
    pu.setup_style()
    fig, axes = plt.subplots(1, len(CTYPES), figsize=(6 * len(CTYPES), 5))

    for ax, ctype in zip(axes, CTYPES):
        sub = df[df['constraint_type'] == ctype]
        ma_grp = sub[sub['angle_strategy'] == 'ma-QAOA'].sort_values('n_layers')
        if ma_grp.empty:
            continue
        ax.plot(ma_grp['n_layers'], ma_grp['optimize_time'],
                marker='s', color=pu.ANGLE_COLORS['ma-QAOA'], linewidth=2)
        ax.set_title(ctype, fontsize=11)
        ax.set_xlabel('ma-QAOA layers (p)')
        ax.set_ylabel('Optimisation time (s)')

    fig.suptitle('VCG: ma-QAOA optimisation time vs depth')
    pu.save_fig(fig, save_path)
    print(f'Saved: {save_path}')


def plot_distributions(df, save_path: str) -> None:
    """Measurement distributions at threshold / best-AR layer (1×2 grid)."""
    pu.setup_style()
    fig, axes = plt.subplots(1, len(CTYPES), figsize=(6 * len(CTYPES), 4))

    for ax, ctype in zip(axes, CTYPES):
        sub = df[(df['constraint_type'] == ctype) & (df['angle_strategy'] == 'ma-QAOA')]
        sub = sub.sort_values('n_layers')
        if sub.empty:
            ax.axis('off')
            continue

        thresh = sub[sub['threshold_reached']]
        row = thresh.iloc[0] if not thresh.empty else sub.loc[sub['AR'].idxmax()]

        counts   = row['counts']
        outcomes = row['outcomes']
        total    = sum(counts.values())
        n_good   = outcomes.count(-1)
        states   = sorted(counts.keys())
        probs    = [counts[s] / total for s in states]
        colors   = [
            pu._ROSE_PINE['foam'] if outcomes[int(s, 2)] == -1
            else pu._ROSE_PINE['love']
            for s in states
        ]

        ax.bar(range(len(states)), probs, color=colors, width=1.0, linewidth=0)
        ax.axhline(1 / n_good, color=pu._ROSE_PINE['gold'],
                   linewidth=1.2, linestyle='--', label='Uniform (1/n_good)')
        ax.set_title(
            f"{ctype}  |  ma-QAOA  p={row['n_layers']}\nAR={row['AR']:.4f}",
            fontsize=9,
        )
        ax.set_xlabel('State index')
        ax.set_xticks([])
        ax.set_ylabel('Probability')
        ax.legend(fontsize=8)

    handles = [
        mpatches.Patch(color=pu._ROSE_PINE['foam'], label='Good state'),
        mpatches.Patch(color=pu._ROSE_PINE['love'], label='Bad state'),
        mpatches.Patch(color=pu._ROSE_PINE['gold'], label='Uniform (1/n_good)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.suptitle('VCG measurement distributions (ma-QAOA at threshold / best layer)', y=1.04)
    pu.save_fig(fig, save_path)
    print(f'Saved: {save_path}')


plot_ar_sweep(df,      'examples/figures/vcg_layer_sweep_ar.png')
plot_time_sweep(df,    'examples/figures/vcg_layer_sweep_time.png')
plot_distributions(df, 'examples/figures/vcg_layer_sweep_distributions.png')

print('\nDone.')
