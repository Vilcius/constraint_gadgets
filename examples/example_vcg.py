"""
example_vcg.py -- Toy demo: VCG training on a single cardinality constraint.

Trains a Variable Constraint Gadget for 'x_0 + x_1 + x_2 == 1' using both
QAOA and ma-QAOA angle strategies.  Results are collected via ResultsCollector
and saved to examples/results/example_vcg_results.pkl.

Run from the project root:
    python examples/example_vcg.py

Output
------
  Prints AR, P(feasible), depth, num_gates for each strategy to stdout.
  Saves measurement distribution plot to examples/figures/vcg_example_counts.png.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from core import vcg as vcg_module
from experiment import ResultsCollector, collect_vcg_data
from analyze_results.plot_feasibility import plot_vcg_counts

os.makedirs('examples/figures', exist_ok=True)
os.makedirs('examples/results', exist_ok=True)

CONSTRAINT = 'x_0 + x_1 + x_2 == 1'   # 3-variable cardinality
N_X = 3
FLAG_WIRE = N_X   # one flag qubit on wire 3


# ══════════════════════════════════════════════════════════════════════════════
# 1. Train VCG for each angle strategy; collect results
# ══════════════════════════════════════════════════════════════════════════════

print(f"Constraint: {CONSTRAINT}\n")

collector = ResultsCollector()
collector.load('examples/results/example_vcg_results.pkl')

rows = []
for strategy in ['QAOA', 'ma-QAOA']:
    print(f"Training VCG [{strategy}] ...")
    gadget = vcg_module.VCG(
        constraints=[CONSTRAINT],
        flag_wires=[FLAG_WIRE],
        angle_strategy=strategy,
        n_layers=1,
        steps=50,
        num_restarts=20,
    )
    row = collect_vcg_data(gadget, constraint_type='cardinality')
    collector.add(row)
    rows.append(row)

    ar      = row['AR'][0]
    p_feas  = sum(v for k, v in row['counts'][0].items() if k[-1] == '0') / sum(row['counts'][0].values())
    depth   = row['resources'][0].depth if row['resources'][0] is not None else 'n/a'
    n_gates = row['resources'][0].num_gates if row['resources'][0] is not None else 'n/a'
    print(f"  AR          = {ar:.4f}")
    print(f"  P(feasible) = {p_feas:.4f}")
    print(f"  Depth       = {depth}")
    print(f"  Num gates   = {n_gates}\n")

collector.save('examples/results/example_vcg_results.pkl')
print("Saved results to examples/results/example_vcg_results.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Plot measurement distributions
# ══════════════════════════════════════════════════════════════════════════════

plot_vcg_counts(
    rows=rows,
    constraint_label=CONSTRAINT,
    save_path='examples/figures/vcg_example_counts.png',
)
print("Saved: examples/figures/vcg_example_counts.png")
print("\nDone.")
