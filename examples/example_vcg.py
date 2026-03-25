"""
example_vcg.py -- Toy demo: VCG training on a single knapsack constraint.

Trains a Variational Constraint Gadget (VCG) for
'3*x_0 + 2*x_1 + x_2 <= 3' using the prescribed two-stage procedure:
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

import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from core.vcg import VCG
from core import constraint_handler as ch

os.makedirs('examples/figures', exist_ok=True)
os.makedirs('examples/results', exist_ok=True)

CONSTRAINT = '3*x_0 + 2*x_1 + x_2 <= 3'   # 3-variable knapsack


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
# 3. Save results
# ══════════════════════════════════════════════════════════════════════════════

results = {
    'constraint': CONSTRAINT,
    'ar': gadget.ar,
    'n_layers': gadget.n_layers,
    'p_feasible': p_feas,
    'train_time': gadget.train_time,
    'opt_angles': gadget.opt_angles,
}
with open('examples/results/example_vcg_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nSaved results to examples/results/example_vcg_results.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plot measurement distribution
# ══════════════════════════════════════════════════════════════════════════════

counts = gadget.do_counts_circuit(shots=10_000)
parsed = ch.parse_constraints([CONSTRAINT])
total = sum(counts.values())

keys = sorted(counts.keys())
probs = [counts[k] / total for k in keys]
feasible = [ch.check_feasibility(k, parsed) for k in keys]
colors = ['steelblue' if f else 'salmon' for f in feasible]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(keys, probs, color=colors, alpha=0.85)
ax.set_title(f'VCG measurement distribution\n{CONSTRAINT}  |  AR={gadget.ar:.3f}  P(feas)={p_feas:.3f}')
ax.set_xlabel('Bitstring')
ax.set_ylabel('Probability')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('examples/figures/vcg_example_counts.png', dpi=150)
plt.close()
print("Saved: examples/figures/vcg_example_counts.png")
print("\nDone.")
