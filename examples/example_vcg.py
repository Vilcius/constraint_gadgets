"""
example_vcg.py -- Toy demo: VCG training on a single cardinality constraint.

Trains a Variational Constraint Gadget (VCG) for
'x_0 + x_1 + x_2 == 1' using the prescribed two-stage procedure:
  Stage 1 -- QAOA p=1 warm-start (2 parameters, fast).
  Stage 2 -- ma-QAOA layer sweep until AR >= ar_threshold or entropy threshold.

P(feasible) is measured by directly evaluating the constraint on output
bitstrings -- no flag qubit is involved.

Run from the project root:
    python examples/example_vcg.py

Output
------
  Prints AR, P(feasible), n_layers, and train_time to stdout.
  Saves measurement distribution plot to examples/figures/vcg_example_counts.png.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from core.vcg import VCG
from analyze_results.results_helper import ResultsCollector, collect_vcg_data
from analyze_results.plot_feasibility import plot_vcg_counts

os.makedirs('examples/figures', exist_ok=True)
os.makedirs('examples/results', exist_ok=True)

CONSTRAINT = 'x_0 + x_1 + x_2 == 1'   # 3-variable cardinality


# ══════════════════════════════════════════════════════════════════════════════
# 1. Create and train a VCG gadget
# ══════════════════════════════════════════════════════════════════════════════

print(f"Constraint: {CONSTRAINT}\n")

print("Training VCG ...")
gadget = VCG(
    constraints=[CONSTRAINT],
    ar_threshold=0.999,
    entropy_threshold=0.9,
    max_layers=8,
    qaoa_restarts=5,
    qaoa_steps=150,
    ma_restarts=20,
    ma_steps=200,
    lr=0.05,
    samples=10_000,
)
gadget.train(verbose=True)

print(f"\nTraining complete.")
print(f"  AR          = {gadget.ar:.4f}")
print(f"  n_layers    = {gadget.n_layers}")
print(f"  Train time  = {gadget.train_time:.1f} s")


# ══════════════════════════════════════════════════════════════════════════════
# 2. P(feasible): constraint check on measured bitstrings (no flag qubit)
# ══════════════════════════════════════════════════════════════════════════════

p_feas = gadget.p_feasible(shots=10_000)
print(f"  P(feasible) = {p_feas:.4f}  (constraint check on 10 000 shots)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Collect results and save
# ══════════════════════════════════════════════════════════════════════════════

collector = ResultsCollector()
collector.load('examples/results/example_vcg_results.pkl')

row = collect_vcg_data(gadget, constraint_type='cardinality')
collector.add(row)

collector.save('examples/results/example_vcg_results.pkl')
print("\nSaved results to examples/results/example_vcg_results.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plot measurement distributions
# ══════════════════════════════════════════════════════════════════════════════

counts = gadget.do_counts_circuit(shots=10_000)
plot_vcg_counts(
    rows=[row],
    constraint_label=CONSTRAINT,
    save_path='examples/figures/vcg_example_counts.png',
)
print("Saved: examples/figures/vcg_example_counts.png")
print("\nDone.")
