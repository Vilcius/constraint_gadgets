# analyze_results

Modular analysis pipeline for constraint_gadget experiments.

## Modules

| File | Purpose |
|---|---|
| `data_loader.py` | Load VCG / HybridQAOA result pickles; filter DataFrames |
| `metrics.py` | P(feasible), P(optimal), AR augmentation, summary stats |
| `plot_utils.py` | Shared light-theme styling (paper-friendly), colour maps, `save_fig` |
| `plot_ar.py` | AR vs n_x, by constraint type, by angle strategy |
| `plot_feasibility.py` | P(feasible) and P(optimal) plots; AR vs P(feasible) scatter |
| `plot_resources.py` | Estimated shots, circuit depth, time breakdown |
| `statistical_tests.py` | Mann-Whitney U (angle strategies), Kruskal-Wallis (families) |
| `main_analysis.py` | CLI entry point — loads, computes, plots, exports |
| `results_helper.py` | `ResultsCollector`, `GadgetDatabase`, `collect_vcg_data`, `collect_hybrid_data`, `collect_penalty_data`, CSV/constraint utilities |
| `../data/make_data.py` | QUBO generation and optimal-x brute force (lives in `data/`) |

## Quick start

```bash
python analyze_results/main_analysis.py \
    --vcg   gadgets/gadget_db.pkl \
    --hybrid results/hybrid_vs_penalty.pkl \
    --output-dir ./analysis_output/
```

Output directories:
- `analysis_output/figures/ar/`
- `analysis_output/figures/feasibility/`
- `analysis_output/figures/resources/`
- `analysis_output/summaries/`
- `analysis_output/statistical_tests/`

## results_helper utilities

| Function / Class | Description |
|---|---|
| `ResultsCollector` | Accumulate experiment rows incrementally; persist/resume from pickle |
| `GadgetDatabase` | Lightweight VCG store: lookup by constraint key for HybridQAOA |
| `read_typed_csv(path)` | Parse `n_vars; [constraint, ...]` CSV format |
| `collect_vcg_data(gadget, ...)` | Extract metrics from a trained VCG instance |
| `collect_hybrid_data(solver, ...)` | Extract metrics from a HybridQAOA instance |
| `collect_penalty_data(solver, ...)` | Extract metrics from a PenaltyQAOA instance |
| `remap_constraint_to_vars(c, vars)` | Embed zero-indexed constraint into QUBO variable positions |
