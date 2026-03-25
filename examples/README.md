# examples

Runnable examples and experiment scripts. All scripts run from the project root.

## Files

| File | Purpose |
|---|---|
| `example_vcg.py` | VCG usage: train a flag-free gadget on a single constraint and inspect its state |
| `example_hybrid.py` | HybridQAOA usage: solve a three-constraint QUBO comparing HybridQAOA vs PenaltyQAOA |

## Running

```bash
# VCG example: train on 3*x_0 + 2*x_1 + x_2 <= 3, print AR / P(feasible), plot counts
python examples/example_vcg.py

# HybridQAOA vs PenaltyQAOA -- three-constraint COP on 7 decision variables
python examples/example_hybrid.py
```

## example_vcg.py

Demonstrates the VCG two-stage training procedure on a 3-variable knapsack
constraint `3*x_0 + 2*x_1 + x_2 <= 3`:

1. Stage 1 -- QAOA p=1 warm-start (2 parameters, fast).
2. Stage 2 -- ma-QAOA layer sweep until AR >= ar_threshold or entropy threshold met.

P(feasible) is computed by evaluating the constraint directly on measured bitstrings
-- no flag or ancilla qubit is used.

**Outputs** (written to `examples/results/` and `examples/figures/`):

| Output | Description |
|---|---|
| `example_vcg_results.pkl` | Collected results row |
| `vcg_example_counts.png` | Measurement distribution plot |

## example_hybrid.py

Compares HybridQAOA against a full-penalisation baseline (PenaltyQAOA) on a
three-constraint combinatorial optimisation problem over 7 binary decision
variables (`x_0 ... x_6`).

**Constraint routing:**

| Label | Constraint | Variables | Handling |
|---|---|---|---|
| A | `x_0 + x_1 + x_2 == 1` | {0, 1, 2} | Structural -- Dicke state prep (exact) |
| B | `6*x_3 + 2*x_4 + 2*x_5 <= 3` | {3, 4, 5} | Structural -- VCG gadget (trained) |
| C | `x_1 + x_4 + x_6 <= 1` | {1, 4, 6} | Penalized (overlaps A and B) |

HybridQAOA handles constraint partitioning internally; no manual flag-wire setup
is needed.

**Outputs** (written to `examples/results/` and `examples/figures/`):

| Output | Description |
|---|---|
| `example_hybrid_results.pkl` | Collected results rows |
| `hybrid_example_metrics.png` | Side-by-side AR / P(feasible) / P(optimal) bar chart |
| `hybrid_example_counts.png` | Top-20 measurement distributions with feasibility colour coding |
