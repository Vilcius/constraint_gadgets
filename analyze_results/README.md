# analyze_results

Modular analysis pipeline for constraint_gadget experiments.

## Modules

| File | Purpose |
|---|---|
| `data_loader.py` | Load VCG / HybridQAOA result pickles; filter DataFrames |
| `metrics.py` | P(feasible), P(optimal), AR augmentation, summary stats |
| `plot_utils.py` | Shared rose-pine styling, colour maps, `save_fig` |
| `plot_ar.py` | AR vs n_x, by constraint type, by angle strategy |
| `plot_feasibility.py` | P(feasible) and P(optimal) plots; AR vs P(feasible) scatter |
| `plot_resources.py` | Estimated shots, circuit depth, time breakdown |
| `statistical_tests.py` | Mann-Whitney U (angle strategies), Kruskal-Wallis (families) |
| `main_analysis.py` | CLI entry point — loads, computes, plots, exports |
| `../data/make_data.py` | QUBO generation and optimal-x brute force (lives in `data/`) |

## Quick start

```bash
python analyze_results/main_analysis.py \
    --vcg   results/cardinality_constraint_results.pkl \
    --hybrid results/hybrid_cardinality_results.pkl \
    --output-dir ./analysis_output/
```

Output directories:
- `analysis_output/figures/ar/`
- `analysis_output/figures/feasibility/`
- `analysis_output/figures/resources/`
- `analysis_output/summaries/`
- `analysis_output/statistical_tests/`
