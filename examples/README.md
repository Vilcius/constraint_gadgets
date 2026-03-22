# examples

Runnable examples and experiment scripts. All scripts run from the project root.

## Files

| File | Purpose |
|---|---|
| `example_vcg.py` | Minimal VCG usage: train a gadget and inspect its state |
| `example_hybrid.py` | Minimal HybridQAOA usage: solve a small constrained QUBO |
| `test_vcg_layers.py` | VCG layer sweep: compare QAOA p=1 baseline vs ma-QAOA sweep on knapsack constraints |
| `vcg_results.md` | Detailed write-up of the VCG layer sweep results and methodology |

## Running

```bash
# Basic VCG example
python examples/example_vcg.py

# Basic HybridQAOA example
python examples/example_hybrid.py

# VCG layer sweep (trains VCGs, saves results + figures; ~15-30 min)
python examples/test_vcg_layers.py
```

## test_vcg_layers.py

Compares QAOA and ma-QAOA as VCG training strategies on two 5-variable constraints:

- `knapsack`: linear inequality `5*x_0 + 10*x_1 + 1*x_2 + 9*x_3 + 6*x_4 <= 19`
- `quad_knapsack`: quadratic inequality over the same variables

**Procedure:**
1. Single QAOA p=1 run (2 parameters, ~8 s) to obtain a warm-start.
2. ma-QAOA layer sweep: p=1 seeded from QAOA angles; p>1 jointly re-optimises
   all layers warm-started from the previous depth. Stops at AR ≥ 0.95.

**Outputs** (written to `examples/results/` and `examples/figures/`):

| Output | Description |
|---|---|
| `vcg_layer_sweep.pkl` | Full results DataFrame |
| `vcg_layer_sweep_ar.png` | AR vs ma-QAOA depth, with QAOA p=1 baseline |
| `vcg_layer_sweep_time.png` | Optimisation time vs depth |
| `vcg_layer_sweep_distributions.png` | Measurement distributions at threshold layer |

See `vcg_results.md` for a detailed analysis of the results.
