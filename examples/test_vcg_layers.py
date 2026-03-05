"""
test_vcg_layers.py -- QAOA vs ma-QAOA VCG layer sweep.

Trains a VCG with increasing layer count until AR >= THRESHOLD or MAX_LAYERS
is reached.  Runs over two constraint types to compare Hamiltonian complexity:
  - Linear knapsack    (5 vars): compact Pauli structure due to dominance
  - Quadratic knapsack (5 vars): dense Pauli structure from cross terms

For each run that reaches the threshold, records the measurement distribution.

Saves:
  examples/results/vcg_layer_sweep.pkl   -- DataFrame with all run data
  examples/figures/vcg_layer_sweep_*.png -- distribution plots

Run from project root:
    python examples/test_vcg_layers.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pennylane import numpy as np

from core import vcg as vcg_module
from analyze_results.results_helper import read_typed_csv

os.makedirs('examples/results', exist_ok=True)
os.makedirs('examples/figures', exist_ok=True)

# ── constraints ────────────────────────────────────────────────────────────────
knap_rows  = read_typed_csv('data/knapsack_constraints.csv')
qknap_rows = read_typed_csv('data/quadratic_knapsack_constraints.csv')

CONSTRAINTS = {
    'knapsack':    next(cs[0] for n, cs in knap_rows  if n == 5),
    'quad_knapsack': next(cs[0] for n, cs in qknap_rows if n == 5),
}

# ── settings ───────────────────────────────────────────────────────────────────
# Reduced for demo speed (~70s/run with 32 Pauli terms).
# For production runs increase STEPS>=200, NUM_RESTARTS>=15.
THRESHOLD    = 0.95
MAX_LAYERS   = 5
STEPS        = 100
NUM_RESTARTS = 5
LR           = 0.01

# VCG Hamiltonian is always binary {-1, +1}: C_min=-1, C_max=+1
def ar_from_cost(cost: float) -> float:
    return (float(cost) - 1.0) / -2.0


# ── sweep ──────────────────────────────────────────────────────────────────────
rows = []

for ctype, constraint in CONSTRAINTS.items():
    # Inspect Hamiltonian structure once
    probe = vcg_module.VCG(
        constraints=[constraint], flag_wires=[5],
        angle_strategy='ma-QAOA', decompose=True, n_layers=1, steps=1, num_restarts=1,
    )
    n_good = probe.outcomes.count(-1)
    n_bad  = probe.outcomes.count(1)
    n_pauli = probe.num_gamma

    print("=" * 70, flush=True)
    print(f"Constraint type : {ctype}", flush=True)
    print(f"Constraint      : {constraint[:80]}", flush=True)
    print(f"States          : {len(probe.outcomes)} total,  {n_good} good,  {n_bad} bad", flush=True)
    print(f"Pauli terms     : {n_pauli}   (wires: {probe.num_beta})", flush=True)
    print(f"Hamiltonian     : {probe.constraint_Ham}", flush=True)
    print("=" * 70, flush=True)

    for angle_strategy in ('QAOA', 'ma-QAOA'):
        print(f"\n  Strategy: {angle_strategy}", flush=True)
        reached = False

        for n_layers in range(1, MAX_LAYERS + 1):
            gadget = vcg_module.VCG(
                constraints=[constraint],
                flag_wires=[5],
                angle_strategy=angle_strategy,
                decompose=True,
                n_layers=n_layers,
                steps=STEPS,
                num_restarts=NUM_RESTARTS,
                learning_rate=LR,
            )

            n_params = (
                n_layers * (gadget.num_gamma + gadget.num_beta)
                if angle_strategy == 'ma-QAOA'
                else n_layers * 2
            )

            opt_cost, _ = gadget.optimize_angles(gadget.do_evolution_circuit)
            ar = ar_from_cost(opt_cost)

            # Get measurement counts at this layer
            counts = gadget.do_counts_circuit(shots=10_000)

            status = '✓' if ar >= THRESHOLD else ' '
            print(
                f"  {status} p={n_layers}: AR={ar:.4f}  params={n_params:3d}"
                f"  time={gadget.optimize_time:.1f}s"
            )

            rows.append({
                'constraint_type': ctype,
                'constraint':      constraint,
                'n_pauli_terms':   n_pauli,
                'n_good_states':   n_good,
                'n_bad_states':    n_bad,
                'angle_strategy':  angle_strategy,
                'n_layers':        n_layers,
                'n_params':        n_params,
                'AR':              ar,
                'optimize_time':   gadget.optimize_time,
                'counts':          counts,
                'outcomes':        gadget.outcomes,
                'threshold_reached': ar >= THRESHOLD,
            })

            if ar >= THRESHOLD:
                reached = True
                break

        if not reached:
            print(f"    Did not reach AR >= {THRESHOLD} within {MAX_LAYERS} layers.", flush=True)

    print(flush=True)

# ── save DataFrame ─────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
save_path = 'examples/results/vcg_layer_sweep.pkl'
df.to_pickle(save_path)
print(f"Saved {len(df)} rows to {save_path}", flush=True)

# ── summary table ──────────────────────────────────────────────────────────────
print(flush=True)
print("Summary (threshold runs only)", flush=True)
print("-" * 75, flush=True)
print(f"  {'Constraint':<16} {'Strategy':<10} {'p*':>3}  {'AR':>7}  "
      f"{'params':>6}  {'Pauli terms':>11}  {'time(s)':>8}", flush=True)
print("  " + "-" * 65, flush=True)
for _, r in df[df['threshold_reached']].iterrows():
    print(
        f"  {r['constraint_type']:<16} {r['angle_strategy']:<10} {r['n_layers']:>3}"
        f"  {r['AR']:>7.4f}  {r['n_params']:>6}  {r['n_pauli_terms']:>11}  "
        f"{r['optimize_time']:>8.1f}",
        flush=True
    )

# ── distribution plots ─────────────────────────────────────────────────────────

def plot_distributions(df_threshold: pd.DataFrame, save_path: str) -> None:
    """Plot measurement distributions for all threshold-reaching runs."""
    runs = list(df_threshold.iterrows())
    n = len(runs)
    if n == 0:
        return

    # Rose-pine colours
    FOAM = '#9ccfd8'   # good states
    LOVE = '#eb6f92'   # bad states
    LINE = '#f6c177'   # equal-superposition reference line

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (_, row) in zip(axes, runs):
        counts  = row['counts']
        outcomes = row['outcomes']
        total   = sum(counts.values())
        n_good  = row['n_good_states']

        # Sort states by index; colour by good/bad
        states  = sorted(counts.keys())
        probs   = [counts[s] / total for s in states]
        colours = [FOAM if outcomes[int(s, 2)] == -1 else LOVE for s in states]

        ax.bar(range(len(states)), probs, color=colours, width=1.0, linewidth=0)
        ax.axhline(1 / n_good, color=LINE, linewidth=1.2, linestyle='--',
                   label=f'Uniform (1/{n_good})')

        ax.set_title(
            f"{row['constraint_type']}  |  {row['angle_strategy']}  p={row['n_layers']}\n"
            f"AR={row['AR']:.4f}",
            fontsize=9
        )
        ax.set_xlabel('State index')
        ax.set_ylabel('Probability')
        ax.set_xticks([])

        good_patch = mpatches.Patch(color=FOAM, label='Good state')
        bad_patch  = mpatches.Patch(color=LOVE, label='Bad state')
        ax.legend(handles=[good_patch, bad_patch,
                            mpatches.Patch(color=LINE, label=f'Uniform 1/{n_good}')],
                  fontsize=7)

    fig.suptitle('VCG measurement distributions (threshold runs)', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}", flush=True)


threshold_df = df[df['threshold_reached']].reset_index(drop=True)
plot_distributions(threshold_df, 'examples/figures/vcg_layer_sweep_distributions.png')
