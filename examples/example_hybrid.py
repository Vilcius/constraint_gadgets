"""
example_hybrid.py -- Three-constraint HybridQAOA vs full PenaltyQAOA.

Problem structure (7 decision variables, x_0 .. x_6)
-----------------------------------------------------
  Constraint A  (structural -- Dicke state prep)
      x_0 + x_1 + x_2 == 1          vars {0, 1, 2}
      Exactly one of three items is selected.
      All unit +1 coefficients and an equality -> Dicke-compatible.
      HybridQAOA prepares this subspace exactly via a log-depth W-state
      circuit and an XY mixer -- no ancilla qubit, zero approximation error.

  Constraint B  (structural -- VCG gadget)
      6*x_3 + 2*x_4 + 2*x_5 <= 3   vars {3, 4, 5}
      Weighted capacity constraint.  Non-unit coefficients and an inequality
      make it NOT Dicke-compatible.  HybridQAOA trains a VCG gadget whose
      ground state is the uniform superposition over feasible assignments,
      then uses it as the initial state and Grover mixer.

  Constraint C  (penalized)
      x_1 + x_4 + x_6 <= 1          vars {1, 4, 6}
      Shared-resource constraint that deliberately overlaps both groups:
      x_1 in A, x_4 in B, x_6 is a free variable.
      Enforced by adding delta*(x_1 + x_4 + x_6 - 1 + s)^2 to the Hamiltonian.

  Variable layout:
      0  1  2  |  3  4  5  |  6        (decision variables)
      <-- A -->  <-- B -->   ^ C only
                    C overlaps: x_1 (A) and x_4 (B)

  QUBO: 7x7 random matrix loaded from data/qubos.csv.

Comparison
----------
  HybridQAOA: A and B structural (exact/gadget), C penalized.
  PenaltyQAOA: all three constraints fully penalized (baseline).
  Both methods run a warm-started layer sweep p = 1 .. MAX_LAYERS.

Run from the project root:
    python examples/example_hybrid.py

Output
------
  Prints per-layer metrics table (AR, P(feasible), P(optimal)) to stdout.
  Saves collected results to examples/results/example_hybrid_results.pkl.
  Saves figures to examples/figures/:
    hybrid_example_layer_sweep.png  --  AR / P(feas) / P(opt) vs layers
    hybrid_example_counts.png       --  measurement distributions at final layer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pandas as pd

from core import constraint_handler as ch
from core import hybrid_qaoa as hq
from core import penalty_qaoa as pq
from core.qaoa_base import ising_hamiltonian_extremes
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results.results_helper import (
    ResultsCollector,
    read_typed_csv, remap_constraint_to_vars,
)
from analyze_results.metrics import compute_comparison_metrics
from analyze_results.plot_feasibility import plot_layer_sweep, plot_outcome_distributions

os.makedirs('examples/figures', exist_ok=True)
os.makedirs('examples/results', exist_ok=True)

N_X        = 7
SHOTS      = 10_000
MAX_LAYERS = 5


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load and configure constraints
# ══════════════════════════════════════════════════════════════════════════════

card_rows = read_typed_csv('data/cardinality_constraints.csv')
knap_rows = read_typed_csv('data/knapsack_constraints.csv')

# Constraint A -- Dicke-compatible (unit coefficients, equality, n=3)
c_a = next(cs[0] for n, cs in card_rows if n == 3 and '== 1' in cs[0])
# c_a = 'x_0 + x_1 + x_2 == 1', embedded on vars {0, 1, 2} (no remapping needed)

# Constraint B -- NOT Dicke-compatible (weighted inequality, n=3)
# Pick first 3-variable knapsack; remap to vars {3, 4, 5} -- disjoint from A.
c_b_raw = next(cs[0] for n, cs in knap_rows if n == 3)
c_b = remap_constraint_to_vars(c_b_raw, [3, 4, 5])
# e.g. '6*x_3 + 2*x_4 + 2*x_5 <= 3'

# Constraint C -- penalized; overlaps A (x_1) and B (x_4), adds x_6
# Use a 3-variable cardinality inequality remapped to {1, 4, 6}.
c_c_raw = next(cs[0] for n, cs in card_rows if n == 3 and '<= 1' in cs[0])
c_c = remap_constraint_to_vars(c_c_raw, [1, 4, 6])
# -> 'x_1 + x_4 + x_6 <= 1'

all_constraints = [c_a, c_b, c_c]

print("=" * 60)
print("Problem: 3-constraint COP on 7 decision variables")
print("=" * 60)
print(f"  A (Dicke,      structural): {c_a}   [vars {{0,1,2}}]")
print(f"  B (VCG,  structural): {c_b}   [vars {{3,4,5}}]")
print(f"  C (penalized):              {c_c}        [vars {{1,4,6}}]")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load a 7-variable QUBO from data/
# ══════════════════════════════════════════════════════════════════════════════

qubos = read_qubos_from_file('qubos.csv', results_dir='data/')
Q = qubos[N_X][0]['Q']
qubo_string = qubos[N_X][0]['qubo_string']
print(f"QUBO ({N_X}x{N_X}): {qubo_string}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Find optimal feasible solution by brute force
# ══════════════════════════════════════════════════════════════════════════════

min_val, optimal_x, total_min = get_optimal_x(Q, all_constraints)
penalty_weight = float(5 + 2 * abs(total_min))

print(f"Optimal feasible QUBO value : {min_val:.4f}")
print(f"Optimal bitstring(s)         : {optimal_x}")
print(f"Penalty weight (delta)       : {penalty_weight:.2f}\n")

parsed = ch.parse_constraints(all_constraints)
structural_indices, penalty_indices = ch.partition_constraints(parsed, strategy='auto')


# ══════════════════════════════════════════════════════════════════════════════
# 4. Layer sweep p = 1 .. MAX_LAYERS
# ══════════════════════════════════════════════════════════════════════════════

print(f"Running layer sweep p=1..{MAX_LAYERS} ...\n")
print(f"  {'p':>2}  {'Method':<14} {'AR':>8} {'P(feas)':>10} {'P(opt)':>9}")
print("  " + "-" * 48)

rows = []
prev_h_angles = None
prev_p_angles = None

for p in range(1, MAX_LAYERS + 1):

    # ── HybridQAOA ────────────────────────────────────────────────
    hybrid = hq.HybridQAOA(
        qubo=Q,
        all_constraints=parsed,
        structural_indices=structural_indices,
        penalty_indices=penalty_indices,
        penalty_pen=penalty_weight,
        angle_strategy='ma-QAOA',
        mixer='Grover',
        n_layers=p,
        steps=50,
        num_restarts=10,
        learning_rate=0.01,
        cqaoa_steps=30,
        cqaoa_num_restarts=10,
        gadget_db_path='gadgets/vcg_db.pkl',
    )
    opt_cost_h, opt_angles_h = hybrid.optimize_angles(prev_layer_angles=prev_h_angles)
    counts_h = hybrid.do_counts_circuit(shots=SHOTS)
    C_min_h, C_max_h = ising_hamiltonian_extremes(hybrid.problem_ham, hybrid.all_wires)
    m_h = compute_comparison_metrics(counts_h, float(opt_cost_h), C_max_h, C_min_h,
                                     all_constraints, N_X, optimal_x)
    rows.append({'method': 'HybridQAOA', 'layer': p, **m_h, 'counts': counts_h})
    prev_h_angles = jnp.array(opt_angles_h)
    print(f"  {p:>2}  {'HybridQAOA':<14} {m_h['AR']:>8.4f} {m_h['p_feasible']:>10.4f} {m_h['p_optimal']:>9.4f}")

    # ── PenaltyQAOA ───────────────────────────────────────────────
    pen_solver = pq.PenaltyQAOA(
        qubo=Q,
        constraints=all_constraints,
        penalty=penalty_weight,
        angle_strategy='ma-QAOA',
        n_layers=p,
        steps=50,
        num_restarts=10,
        learning_rate=0.01,
    )
    opt_cost_p, opt_angles_p = pen_solver.optimize_angles(prev_layer_angles=prev_p_angles)
    counts_p = pen_solver.do_counts_circuit(shots=SHOTS)
    C_min_p, C_max_p = ising_hamiltonian_extremes(pen_solver.full_Ham, pen_solver.all_wires)
    m_p = compute_comparison_metrics(counts_p, float(opt_cost_p), C_max_p, C_min_p,
                                     all_constraints, N_X, optimal_x)
    rows.append({'method': 'PenaltyQAOA', 'layer': p, **m_p, 'counts': counts_p})
    prev_p_angles = jnp.array(opt_angles_p)
    print(f"  {p:>2}  {'PenaltyQAOA':<14} {m_p['AR']:>8.4f} {m_p['p_feasible']:>10.4f} {m_p['p_optimal']:>9.4f}")

print()

# Save all layer rows
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'counts'} for r in rows])
collector = ResultsCollector()
collector.load('examples/results/example_hybrid_results.pkl')
for r in rows:
    collector.add(r)
collector.save('examples/results/example_hybrid_results.pkl')
print("Saved results to examples/results/example_hybrid_results.pkl\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Plot 1 -- Layer sweep: AR / P(feas) / P(opt) vs p
# ══════════════════════════════════════════════════════════════════════════════

plot_layer_sweep(
    df,
    title='HybridQAOA vs PenaltyQAOA — 3-constraint COP, layer sweep',
    save_path='examples/figures/hybrid_example_layer_sweep.png',
)
print("Saved: examples/figures/hybrid_example_layer_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Plot 2 -- Measurement distributions at final layer
# ══════════════════════════════════════════════════════════════════════════════

final_h = next(r for r in reversed(rows) if r['method'] == 'HybridQAOA')
final_p = next(r for r in reversed(rows) if r['method'] == 'PenaltyQAOA')

plot_outcome_distributions(
    counts={'HybridQAOA': final_h['counts'], 'PenaltyQAOA': final_p['counts']},
    constraints=all_constraints,
    n_x=N_X,
    optimal_x=optimal_x,
    top_n=20,
    structural_constraints=[all_constraints[i] for i in structural_indices],
    penalty_constraints=[all_constraints[i] for i in penalty_indices],
    title=f'Measurement distributions at p={MAX_LAYERS}: decision variables only',
    save_path='examples/figures/hybrid_example_counts.png',
)
print("Saved: examples/figures/hybrid_example_counts.png")
print("\nDone.")
