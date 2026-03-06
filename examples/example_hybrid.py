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
      make it NOT Dicke-compatible.  HybridQAOA trains a Variational Constraint
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
  Saves collected results to examples/results/example_hybrid_results.pkl.
  Saves two figures to examples/figures/:
    hybrid_example_metrics.png  –  side-by-side metric comparison
    hybrid_example_counts.png   –  measurement distributions (top outcomes)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from core import constraint_handler as ch
from core import hybrid_qaoa as hq
from core import penalty_qaoa as pq
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results.results_helper import (
    ResultsCollector,
    read_typed_csv, remap_constraint_to_vars,
    collect_hybrid_data, collect_penalty_data,
)
from analyze_results.metrics import compute_comparison_metrics
from analyze_results.plot_feasibility import plot_method_comparison, plot_outcome_distributions

os.makedirs('examples/figures', exist_ok=True)
os.makedirs('examples/results', exist_ok=True)

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
)

collector = ResultsCollector()
collector.load('examples/results/example_hybrid_results.pkl')

row_h = collect_hybrid_data(
    all_constraints, hybrid, qubo_string,
    min_val=min_val, constraint_type='mixed',
)
collector.add(row_h)
print(f"  Done. AR = {row_h['AR'][0]:.4f}\n")


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

row_p = collect_penalty_data(
    all_constraints, penalty_solver, qubo_string,
    min_val=min_val, constraint_type='mixed',
)
collector.add(row_p)
print(f"  Done. AR = {row_p['AR'][0]:.4f}\n")

collector.save('examples/results/example_hybrid_results.pkl')
print("Saved results to examples/results/example_hybrid_results.pkl\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Metrics
# ══════════════════════════════════════════════════════════════════════════════

m_hybrid  = compute_comparison_metrics(
    row_h['counts'][0], row_h['opt_cost'][0],
    row_h['C_max'][0], row_h['C_min'][0],
    all_constraints, N_X, optimal_x,
)
m_penalty = compute_comparison_metrics(
    row_p['counts'][0], row_p['opt_cost'][0],
    row_p['C_max'][0], row_p['C_min'][0],
    all_constraints, N_X, optimal_x,
)

print("Results:")
print(f"  {'Method':<16} {'AR':>8} {'P(feasible)':>12} {'P(optimal)':>11}")
print("  " + "-" * 52)
for name, m in [("HybridQAOA", m_hybrid), ("PenaltyQAOA", m_penalty)]:
    print(f"  {name:<16} {m['AR']:>8.4f} {m['p_feasible']:>12.4f} {m['p_optimal']:>11.4f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plot 1 – Metric comparison bar chart
# ══════════════════════════════════════════════════════════════════════════════

plot_method_comparison(
    {'HybridQAOA': m_hybrid, 'PenaltyQAOA': m_penalty},
    title='HybridQAOA vs PenaltyQAOA – 3-constraint COP',
    save_path='examples/figures/hybrid_example_metrics.png',
)
print("Saved: examples/figures/hybrid_example_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Plot 2 – Measurement distributions (top outcomes)
# ══════════════════════════════════════════════════════════════════════════════

plot_outcome_distributions(
    counts={'HybridQAOA': row_h['counts'][0], 'PenaltyQAOA': row_p['counts'][0]},
    constraints=all_constraints,
    n_x=N_X,
    optimal_x=optimal_x,
    top_n=20,
    title='Measurement distributions: decision variables only',
    save_path='examples/figures/hybrid_example_counts.png',
)
print("Saved: examples/figures/hybrid_example_counts.png")
print("\nDone.")
