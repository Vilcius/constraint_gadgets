"""
test_vcg_layers.py -- QAOA vs ma-QAOA VCG layer sweep.

Trains a VCG with increasing QAOA depth until AR >= THRESHOLD or MAX_LAYERS
is reached.  Two key optimisation improvements over a naive sweep:

  1. Layer-freezing warm start – when going from p to p+1 layers the first p
     layers' angles are frozen; only the new (p+1)-th layer is optimised.
     This keeps the active parameter count at k (one layer) rather than (p+1)k.

  2. QAOA-seeded initialisation for ma-QAOA – the QAOA sweep runs first and
     stores its optimal angles at every depth.  For ma-QAOA:
       • p=1  : first restart starts from the QAOA p=1 solution (broadcast
                gamma → all num_gamma entries, beta → all num_beta entries).
       • p>1  : as above, but the new-layer warm-start broadcasts the QAOA
                best angles at that depth's new layer.
     Subsequent restarts are random, providing diversity.  This guarantees
     ma-QAOA starts from a point at least as good as QAOA and only improves.

  3. Budget scaled by parameter count – ma-QAOA optimises 38 params per layer
     vs QAOA's 2, so it gets proportionally more restarts and steps.

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
from core import qaoa_base as base
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
LR         = 0.05   # Adam learning rate (larger → faster escape from flat regions)
SHOTS      = 10_000 # measurement shots for distributions
FLAG_WIRE  = 5      # flag qubit index (one per 5-var constraint)

# Per-strategy budgets: ma-QAOA has ~19× more parameters per layer than QAOA,
# so it gets more restarts and steps to give the optimiser a fair chance.
RESTARTS = {'QAOA': 5,  'ma-QAOA': 20}
STEPS    = {'QAOA': 150, 'ma-QAOA': 200}

RESULTS_PATH = 'examples/results/vcg_layer_sweep.pkl'


# ══════════════════════════════════════════════════════════════════════════════
# 2. Layer-freezing optimiser
# ══════════════════════════════════════════════════════════════════════════════

def optimize_new_layer(gadget: vcg_module.VCG,
                       frozen_angles: np.ndarray,
                       starting_new_angles: np.ndarray = None) -> tuple:
    """Freeze all previous layer angles; optimise only the new (last) layer.

    Parameters
    ----------
    gadget : VCG
        Must already be built with n_layers = p+1.
    frozen_angles : np.ndarray
        Optimal angles from the p-layer run, shape (p, params_per_layer).
    starting_new_angles : np.ndarray or None
        Optional warm-start for the first restart, shape (1, params_per_layer).
        Subsequent restarts are always random.  Pass the QAOA new-layer angles
        (broadcast to ma-QAOA format) to seed ma-QAOA from a strong prior.

    Returns
    -------
    best_cost : float
    full_angles : np.ndarray, shape (p+1, params_per_layer)
    """
    params_per_layer = (
        gadget.num_gamma + gadget.num_beta
        if gadget.angle_strategy == 'ma-QAOA'
        else 2
    )
    frozen = np.array(frozen_angles.flatten(), requires_grad=False)

    def cost_fn(new_angles):
        full = np.concatenate([frozen, new_angles.flatten()])
        return gadget.do_evolution_circuit(
            full.reshape(gadget.n_layers, params_per_layer)
        )

    best_cost, new_layer_angles, wall_time = base.run_optimization(
        cost_fn=cost_fn,
        n_layers=1,
        num_gamma=gadget.num_gamma,
        num_beta=gadget.num_beta,
        angle_strategy=gadget.angle_strategy,
        steps=gadget.steps,
        num_restarts=gadget.num_restarts,
        learning_rate=gadget.learning_rate,
        starting_angles=starting_new_angles,
    )

    full_angles = np.concatenate([frozen, new_layer_angles.flatten()])
    full_angles = full_angles.reshape(gadget.n_layers, params_per_layer)
    gadget.opt_angles = full_angles
    gadget.optimize_time = wall_time
    return best_cost, full_angles


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sweep
# ══════════════════════════════════════════════════════════════════════════════

# Start fresh — do not load a previous run with different settings.
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

    # QAOA angles stored per depth for seeding ma-QAOA warm start.
    # qaoa_angles_by_layer[p] has shape (p, 2).
    qaoa_angles_by_layer = {}

    for angle_strategy in ('QAOA', 'ma-QAOA'):
        print(f'\n  Strategy: {angle_strategy}  '
              f'(restarts={RESTARTS[angle_strategy]}, steps={STEPS[angle_strategy]})',
              flush=True)
        reached = False
        prev_best_angles = None

        for n_layers in range(1, MAX_LAYERS + 1):
            gadget = vcg_module.VCG(
                constraints=[constraint],
                flag_wires=[FLAG_WIRE],
                angle_strategy=angle_strategy,
                decompose=(angle_strategy == 'ma-QAOA'),
                n_layers=n_layers,
                steps=STEPS[angle_strategy],
                num_restarts=RESTARTS[angle_strategy],
                learning_rate=LR,
            )

            n_params = (
                n_layers * (gadget.num_gamma + gadget.num_beta)
                if angle_strategy == 'ma-QAOA'
                else n_layers * 2
            )

            if angle_strategy == 'QAOA':
                # ── standard QAOA sweep ───────────────────────────────────
                # Re-optimise ALL layers jointly at every depth: with only
                # 2 params per layer the full parameter space stays small
                # (16 params at p=8) and freezing previous layers would
                # unnecessarily constrain the search.  prev_layer_angles
                # seeds every restart at the previous optimum + random new
                # layer, then Adam is free to adjust all layers together.
                opt_cost, _ = gadget.optimize_angles(
                    gadget.do_evolution_circuit,
                    prev_layer_angles=prev_best_angles,   # None at p=1
                )
                qaoa_angles_by_layer[n_layers] = gadget.opt_angles  # shape (p, 2)

            else:
                # ── ma-QAOA sweep with QAOA-seeded warm start ─────────────
                if prev_best_angles is None:
                    # p=1: broadcast QAOA p=1 optimal angles to ma-QAOA shape
                    qaoa_seed = qaoa_angles_by_layer.get(1)  # shape (1, 2)
                    opt_cost, _ = gadget.optimize_angles(
                        gadget.do_evolution_circuit,
                        starting_angles_from_qaoa=qaoa_seed,
                    )
                else:
                    # p>1: freeze previous layers; warm-start new layer from
                    # the corresponding QAOA depth's last-layer angles.
                    starting_new = None
                    if n_layers in qaoa_angles_by_layer:
                        # Take the new layer slice from QAOA and broadcast.
                        last_qaoa = qaoa_angles_by_layer[n_layers][-1:, :]  # (1,2)
                        starting_new = base.convert_qaoa_to_ma_angles(
                            last_qaoa, gadget.num_gamma, gadget.num_beta, 1
                        )  # (1, num_gamma + num_beta)
                    opt_cost, _ = optimize_new_layer(
                        gadget, prev_best_angles, starting_new_angles=starting_new
                    )

            prev_best_angles = gadget.opt_angles
            ar = (float(opt_cost) - 1.0) / -2.0

            status = '✓' if ar >= THRESHOLD else ' '
            print(
                f'  {status} p={n_layers}: AR={ar:.4f}  params={n_params:3d}'
                f'  time={gadget.optimize_time:.1f}s',
                flush=True,
            )

            row = collect_vcg_data(
                gadget,
                constraint_type=ctype,
                skip_optimize=True,
                shots=SHOTS,
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
# 4. Save
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
# 5. Summary table
# ══════════════════════════════════════════════════════════════════════════════

print('Summary (all runs)')
print('-' * 76)
print(f"  {'Constraint':<16} {'Strategy':<10} {'p':>2}  {'AR':>7}  "
      f"{'params':>6}  {'Pauli terms':>11}  {'time(s)':>8}")
print('  ' + '-' * 66)
for _, r in df.sort_values(['constraint_type', 'angle_strategy', 'n_layers']).iterrows():
    mark = '✓' if r['threshold_reached'] else ' '
    print(
        f"  {mark} {r['constraint_type']:<16} {r['angle_strategy']:<10}"
        f" {r['n_layers']:>2}  {r['AR']:>7.4f}  {r['n_params']:>6}"
        f"  {r['num_gamma']:>11}  {r['optimize_time']:>8.1f}",
    )
print()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Plots
# ══════════════════════════════════════════════════════════════════════════════

CTYPES     = list(CONSTRAINTS.keys())
STRATEGIES = ['QAOA', 'ma-QAOA']


def plot_ar_sweep(df, save_path: str) -> None:
    """AR vs QAOA depth, one panel per constraint type."""
    pu.setup_style()
    fig, axes = plt.subplots(1, len(CTYPES), figsize=(6 * len(CTYPES), 5), sharey=True)

    for ax, ctype in zip(axes, CTYPES):
        sub = df[df['constraint_type'] == ctype]
        for strategy in STRATEGIES:
            grp = sub[sub['angle_strategy'] == strategy].sort_values('n_layers')
            if grp.empty:
                continue
            color = pu.ANGLE_COLORS[strategy]
            ax.plot(grp['n_layers'], grp['AR'],
                    marker='o', color=color, label=strategy, linewidth=2)

        ax.axhline(THRESHOLD, color=pu._ROSE_PINE['gold'],
                   linestyle='--', linewidth=1.2, label=f'Threshold {THRESHOLD}')
        ax.set_title(ctype, fontsize=11)
        ax.set_xlabel('Layers (p)')
        ax.set_ylim(0.5, 1.05)
        ax.set_xticks(range(1, MAX_LAYERS + 1))
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Approximation Ratio (AR)')
    fig.suptitle('VCG: AR vs QAOA depth  '
                 '(layer-freezing + QAOA-seeded ma-QAOA warm start)')
    pu.save_fig(fig, save_path)
    print(f'Saved: {save_path}')


def plot_time_sweep(df, save_path: str) -> None:
    """Optimisation time vs QAOA depth, one panel per constraint type."""
    pu.setup_style()
    fig, axes = plt.subplots(1, len(CTYPES), figsize=(6 * len(CTYPES), 5))

    for ax, ctype in zip(axes, CTYPES):
        sub = df[df['constraint_type'] == ctype]
        for strategy in STRATEGIES:
            grp = sub[sub['angle_strategy'] == strategy].sort_values('n_layers')
            if grp.empty:
                continue
            color = pu.ANGLE_COLORS[strategy]
            ax.plot(grp['n_layers'], grp['optimize_time'],
                    marker='s', color=color, label=strategy, linewidth=2)

        ax.set_title(ctype, fontsize=11)
        ax.set_xlabel('Layers (p)')
        ax.set_ylabel('Optimisation time (s)')
        ax.set_xticks(range(1, MAX_LAYERS + 1))
        ax.legend(fontsize=9)

    fig.suptitle('VCG: optimisation time vs depth  '
                 '(layer-freezing + QAOA-seeded ma-QAOA warm start)')
    pu.save_fig(fig, save_path)
    print(f'Saved: {save_path}')


def plot_distributions(df, save_path: str) -> None:
    """Measurement distributions at threshold / best-AR layer (2×2 grid)."""
    pu.setup_style()
    fig, axes = plt.subplots(
        len(CTYPES), len(STRATEGIES),
        figsize=(6 * len(STRATEGIES), 4 * len(CTYPES)),
    )

    for i, ctype in enumerate(CTYPES):
        sub = df[df['constraint_type'] == ctype]
        for j, strategy in enumerate(STRATEGIES):
            ax = axes[i][j]
            grp = sub[sub['angle_strategy'] == strategy].sort_values('n_layers')
            if grp.empty:
                ax.axis('off')
                continue

            # Threshold row if reached, else best-AR layer
            thresh = grp[grp['threshold_reached']]
            row = thresh.iloc[0] if not thresh.empty else grp.loc[grp['AR'].idxmax()]

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
                       linewidth=1.2, linestyle='--')
            ax.set_title(
                f"{ctype}  |  {strategy}  p={row['n_layers']}\n"
                f"AR={row['AR']:.4f}",
                fontsize=9,
            )
            ax.set_xlabel('State index')
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel('Probability')

    handles = [
        mpatches.Patch(color=pu._ROSE_PINE['foam'], label='Good state'),
        mpatches.Patch(color=pu._ROSE_PINE['love'], label='Bad state'),
        mpatches.Patch(color=pu._ROSE_PINE['gold'], label='Uniform (1/n_good)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.suptitle('VCG measurement distributions (threshold / best layer)', y=1.04)
    pu.save_fig(fig, save_path)
    print(f'Saved: {save_path}')


plot_ar_sweep(df,      'examples/figures/vcg_layer_sweep_ar.png')
plot_time_sweep(df,    'examples/figures/vcg_layer_sweep_time.png')
plot_distributions(df, 'examples/figures/vcg_layer_sweep_distributions.png')

print('\nDone.')
