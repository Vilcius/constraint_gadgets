"""
example_hybrid.py -- Three-constraint HybridQAOA vs full PenaltyQAOA.

Problem structure (7 decision variables, x_0 .. x_6)
-----------------------------------------------------
  Constraint A  (structural – Dicke state prep)
      x_0 + x_1 + x_2 == 1          vars {0, 1, 2}
      Exactly one of three items is selected.
      All unit +1 coefficients and an equality → Dicke-compatible.
      HybridQAOA prepares this subspace exactly via a log-depth W-state
      circuit and an XY mixer – no flag qubit, zero approximation error.

  Constraint B  (structural – VCG gadget)
      6*x_3 + 2*x_4 + 2*x_5 <= 3   vars {3, 4, 5}
      Weighted capacity constraint.  Non-unit coefficients and an inequality
      make it NOT Dicke-compatible.  HybridQAOA trains a Variable Constraint
      Gadget (VCG) whose ground state is the uniform superposition over
      feasible assignments, then uses it as the initial state and Grover
      mixer – one flag qubit (wire 7) marks (un)satisfying assignments.

  Constraint C  (penalized)
      x_1 + x_4 + x_6 <= 1          vars {1, 4, 6}
      Shared-resource constraint that deliberately overlaps both groups:
      x_1 ∈ A, x_4 ∈ B, x_6 is a free variable.
      Enforced by adding δ·(x_1 + x_4 + x_6 – 1 + s)² to the Hamiltonian.

  Variable layout:
      0  1  2  |  3  4  5  |  6        (decision variables)
      ←── A ──→  ←── B ──→   ↑ C only
                    C overlaps: x_1 (A) and x_4 (B)

  QUBO: 7 × 7 random matrix loaded from data/qubos.csv.

Comparison
----------
  HybridQAOA: A and B structural (exact/gadget), C penalized.
  PenaltyQAOA: all three constraints fully penalized (baseline).

Run from the project root:
    python examples/example_hybrid.py

Output
------
  Prints metrics table (AR, P(feasible), P(optimal)) to stdout.
  Saves two figures to examples/figures/:
    hybrid_example_metrics.png  –  side-by-side metric comparison
    hybrid_example_counts.png   –  measurement distributions (top outcomes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pennylane as qml

from core import constraint_handler as ch
from core import hybrid_qaoa as hq
from core import penalty_qaoa as pq
from data.make_data import read_qubos_from_file, get_optimal_x
from run.run_utils import read_typed_csv, remap_constraint_to_vars
from analyze_results.plot_utils import setup_style, _ROSE_PINE, save_fig

os.makedirs('examples/figures', exist_ok=True)

N_X = 7          # QUBO / decision-variable count
SHOTS = 10_000   # measurement shots


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load and configure constraints
# ══════════════════════════════════════════════════════════════════════════════

card_rows = read_typed_csv('data/cardinality_constraints.csv')
knap_rows = read_typed_csv('data/knapsack_constraints.csv')

# Constraint A – Dicke-compatible (unit coefficients, equality, n=3)
c_a = next(cs[0] for n, cs in card_rows if n == 3 and '== 1' in cs[0])
# c_a = 'x_0 + x_1 + x_2 == 1', embedded on vars {0, 1, 2} (no remapping needed)

# Constraint B – NOT Dicke-compatible (weighted inequality, n=3)
# Pick first 3-variable knapsack; remap to vars {3, 4, 5} – disjoint from A.
c_b_raw = next(cs[0] for n, cs in knap_rows if n == 3)
c_b = remap_constraint_to_vars(c_b_raw, [3, 4, 5])
# e.g. '6*x_3 + 2*x_4 + 2*x_5 <= 3'

# Constraint C – penalized; overlaps A (x_1) and B (x_4), adds x_6
# Use a 3-variable cardinality inequality remapped to {1, 4, 6}.
c_c_raw = next(cs[0] for n, cs in card_rows if n == 3 and '<= 1' in cs[0])
c_c = remap_constraint_to_vars(c_c_raw, [1, 4, 6])
# → 'x_1 + x_4 + x_6 <= 1'

all_constraints = [c_a, c_b, c_c]

print("=" * 60)
print("Problem: 3-constraint COP on 7 decision variables")
print("=" * 60)
print(f"  A (Dicke,    structural): {c_a}   [vars {{0,1,2}}]")
print(f"  B (VCG,      structural): {c_b}   [vars {{3,4,5}}]")
print(f"  C (penalized):            {c_c}        [vars {{1,4,6}}]")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load a 7-variable QUBO from data/
# ══════════════════════════════════════════════════════════════════════════════

qubos = read_qubos_from_file('qubos.csv', results_dir='data/')
Q = qubos[N_X][0]['Q']
qubo_string = qubos[N_X][0]['qubo_string']
print(f"QUBO ({N_X}×{N_X}): {qubo_string}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Find optimal feasible solution by brute force
# ══════════════════════════════════════════════════════════════════════════════

min_val, optimal_x, total_min = get_optimal_x(Q, all_constraints)
penalty_weight = float(5 + 2 * abs(total_min))

print(f"Optimal feasible QUBO value : {min_val:.4f}")
print(f"Optimal bitstring(s)         : {optimal_x}")
print(f"Penalty weight (δ)           : {penalty_weight:.2f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HybridQAOA
#    – A enforced via Dicke state prep (exact, no flag qubit)
#    – B enforced via VCG gadget (trained from scratch, 1 flag qubit at wire 7)
#    – C penalized (slack variable added automatically)
#    – Grover mixer reflects about the composed state-prep circuit
# ══════════════════════════════════════════════════════════════════════════════

print("Running HybridQAOA ...")
parsed = ch.parse_constraints(all_constraints)
hybrid = hq.HybridQAOA(
    qubo=Q,
    all_constraints=parsed,
    structural_indices=[0, 1],   # A (Dicke) + B (VCG)
    penalty_indices=[2],         # C penalized
    penalty_str=[penalty_weight],
    penalty_pen=penalty_weight,
    angle_strategy='ma-QAOA',
    mixer='Grover',
    n_layers=1,
    steps=50,
    num_restarts=10,
    cqaoa_steps=30,
    cqaoa_num_restarts=10,
    pre_made=False,
)
opt_cost_h, counts_h, _ = hybrid.solve()

eigvals_h = qml.eigvals(hybrid.problem_ham)
C_max_h, C_min_h = float(max(eigvals_h)), float(min(eigvals_h))
ar_h = (float(opt_cost_h) - C_max_h) / (C_min_h - C_max_h)
print(f"  Done. AR = {ar_h:.4f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PenaltyQAOA (full-penalization baseline)
#    All three constraints converted to quadratic penalty terms.
#    No structured state prep; starts from |+>^n on all qubits.
# ══════════════════════════════════════════════════════════════════════════════

print("Running PenaltyQAOA (full penalization baseline) ...")
penalty_solver = pq.PenaltyQAOA(
    qubo=Q,
    constraints=all_constraints,
    penalty=penalty_weight,
    angle_strategy='ma-QAOA',
    n_layers=1,
    steps=50,
    num_restarts=10,
)
opt_cost_p, _ = penalty_solver.optimize_angles(penalty_solver.do_evolution_circuit)
counts_p = penalty_solver.do_counts_circuit(shots=SHOTS)

eigvals_p = qml.eigvals(penalty_solver.full_Ham)
C_max_p, C_min_p = float(max(eigvals_p)), float(min(eigvals_p))
ar_p = (float(opt_cost_p) - C_max_p) / (C_min_p - C_max_p)
print(f"  Done. AR = {ar_p:.4f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Metrics
# ══════════════════════════════════════════════════════════════════════════════

def is_feasible(bitstring: str, n_x: int = N_X) -> bool:
    """Check first n_x bits against all three constraints."""
    vd = {f'x_{i}': int(bitstring[i]) for i in range(n_x)}
    return all(eval(c, {"__builtins__": {}}, vd) for c in all_constraints)


def aggregate_counts(counts: dict, n_x: int = N_X) -> dict:
    """Collapse auxiliary bits; return {decision_bitstring: probability}."""
    total = sum(counts.values())
    agg: dict = {}
    for bs, cnt in counts.items():
        key = bs[:n_x]
        agg[key] = agg.get(key, 0) + cnt / total
    return agg


def compute_metrics(counts: dict, opt_cost: float,
                    ham: qml.Hamiltonian, C_max: float, C_min: float,
                    n_x: int = N_X) -> dict:
    agg = aggregate_counts(counts, n_x)
    p_feas = sum(p for bs, p in agg.items() if is_feasible(bs))
    p_opt = (sum(p for bs, p in agg.items()
                 if bs in (optimal_x or []))
             if optimal_x else float('nan'))
    ar = (float(opt_cost) - C_max) / (C_min - C_max)
    return dict(AR=ar, p_feasible=p_feas, p_optimal=p_opt)


m_hybrid  = compute_metrics(counts_h, opt_cost_h, hybrid.problem_ham,
                             C_max_h, C_min_h)
m_penalty = compute_metrics(counts_p, opt_cost_p, penalty_solver.full_Ham,
                             C_max_p, C_min_p)

print("Results:")
print(f"  {'Method':<16} {'AR':>8} {'P(feasible)':>12} {'P(optimal)':>11}")
print("  " + "-" * 52)
for name, m in [("HybridQAOA", m_hybrid), ("PenaltyQAOA", m_penalty)]:
    print(f"  {name:<16} {m['AR']:>8.4f} {m['p_feasible']:>12.4f} {m['p_optimal']:>11.4f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plot 1 – Metric comparison bar chart
# ══════════════════════════════════════════════════════════════════════════════

setup_style()
fig, ax = plt.subplots(figsize=(8, 5))

metrics  = ['AR', 'p_feasible', 'p_optimal']
labels   = ['Approximation Ratio', 'P(feasible)', 'P(optimal)']
x        = np.arange(len(metrics))
width    = 0.32
colors   = [_ROSE_PINE['pine'], _ROSE_PINE['love']]

vals_h = [m_hybrid[m]  for m in metrics]
vals_p = [m_penalty[m] for m in metrics]

bars_h = ax.bar(x - width / 2, vals_h, width, label='HybridQAOA',
                color=colors[0], alpha=0.85)
bars_p = ax.bar(x + width / 2, vals_p, width, label='PenaltyQAOA',
                color=colors[1], alpha=0.85)

for bar in list(bars_h) + list(bars_p):
    h = bar.get_height()
    if not np.isnan(h):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Value')
ax.set_title('HybridQAOA vs PenaltyQAOA – 3-constraint COP')
ax.legend()

save_fig(fig, 'examples/figures/hybrid_example_metrics.png')
print("Saved: examples/figures/hybrid_example_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Plot 2 – Measurement distributions (top outcomes)
# ══════════════════════════════════════════════════════════════════════════════

TOP_N = 20

def top_outcomes(counts: dict, n: int = TOP_N, n_x: int = N_X):
    agg = aggregate_counts(counts, n_x)
    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:n]

def bar_colors(outcomes: list) -> list:
    """Green=feasible+optimal, blue=feasible, red=infeasible."""
    out = []
    for bs, _ in outcomes:
        if optimal_x and bs in optimal_x:
            out.append(_ROSE_PINE['foam'])       # optimal
        elif is_feasible(bs):
            out.append(_ROSE_PINE['pine'])       # feasible only
        else:
            out.append(_ROSE_PINE['love'])       # infeasible
    return out


fig2, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, name, top in [
    (axes[0], 'HybridQAOA',  top_outcomes(counts_h)),
    (axes[1], 'PenaltyQAOA', top_outcomes(counts_p)),
]:
    bstrings = [bs for bs, _ in top]
    probs    = [p  for _, p  in top]
    colors_b = bar_colors(top)

    ax.bar(range(len(bstrings)), probs, color=colors_b, alpha=0.85)
    ax.set_xticks(range(len(bstrings)))
    ax.set_xticklabels(bstrings, rotation=90, fontsize=7)
    ax.set_xlabel('Bitstring (decision variables x_0..x_6)')
    ax.set_ylabel('Probability')
    ax.set_title(f'{name} – top {TOP_N} outcomes')

    p_f = sum(p for bs, p in top if is_feasible(bs))
    ax.text(0.98, 0.97, f'P(feas) shown: {p_f:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)

legend_patches = [
    mpatches.Patch(color=_ROSE_PINE['foam'], label='Optimal'),
    mpatches.Patch(color=_ROSE_PINE['pine'], label='Feasible'),
    mpatches.Patch(color=_ROSE_PINE['love'], label='Infeasible'),
]
fig2.legend(handles=legend_patches, loc='upper center', ncol=3,
            bbox_to_anchor=(0.5, 1.02))
fig2.suptitle('Measurement distributions: decision variables only', y=1.05)

save_fig(fig2, 'examples/figures/hybrid_example_counts.png')
print("Saved: examples/figures/hybrid_example_counts.png")
print("\nDone.")
