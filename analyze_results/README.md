# analyze_results

Modular analysis pipeline for constraint_gadget experiments.

## Modules

| File | Purpose |
|---|---|
| `split_results.py` | Split raw merged pkl into typed DataFrames (comparison_ar, comparison_resources, etc.) |
| `compute_circuit_resources.py` | Analytical gate-count table for all experiments |
| `compute_vcg_resources.py` | Analytical gate-count table for trained VCGs |
| `generate_plots.py` | Top-level orchestrator: combine splits, run analysis, copy plots and stats to paper/ |
| `generate_results_markdown.py` | GitHub-readable markdown tables of raw results → `results/*/README.md` |
| `main_analysis.py` | Core analysis: produces all figures, summaries, and statistical tests |
| `build_problem_table.py` | Problem metadata table from raw results (called automatically during merge) |
| `results_helper.py` | `ResultsCollector`, CSV parsing, constraint remapping utilities |
| `metrics.py` | P(feasible), P(optimal), AR_feas, summary stats |
| `plot_utils.py` | Shared matplotlib styling (rose-pine palette), `save_fig` |
| `plot_ar.py` | AR and AR_feas plots vs n_x, by constraint type, by angle strategy |
| `plot_feasibility.py` | P(feasible) and P(optimal) plots |
| `plot_resources.py` | Estimated shots, circuit depth, time breakdown |
| `plot_vcg_db.py` | VCG database summary plots (entropy, layers, convergence) |
| `statistical_tests.py` | Mann-Whitney U (angle strategies), Kruskal-Wallis (families) |

## Quick start

The normal entry point is `generate_plots.py`, which combines both splits,
runs the full analysis, and copies outputs to `paper/`.

```bash
# 1. Split raw results (once per split, after experiments complete)
python analyze_results/split_results.py \
    --pc-qaoa results/overlapping/pc_qaoa_vs_penalty.pkl \
    --output-dir results/overlapping/

python analyze_results/split_results.py \
    --pc-qaoa results/disjoint/pc_qaoa_vs_penalty.pkl \
    --output-dir results/disjoint/

# 2. Compute circuit resources
python analyze_results/compute_circuit_resources.py
python analyze_results/compute_vcg_resources.py

# 3. Combine, plot, and export
python analyze_results/generate_plots.py
```

Output layout:
```
analysis_output/
    combined/figures/ar/
    combined/figures/feasibility/
    combined/figures/resources/
    combined/figures/vcg_db/
    combined/summaries/
    combined/statistical_tests/
    disjoint/   ← per-split figures and summaries
    overlapping/
```

`generate_plots.py` also writes `results/{disjoint,overlapping}/README.md`
with GitHub-readable tables of every problem instance and its results.

## results_helper utilities

| Function / Class | Description |
|---|---|
| `ResultsCollector` | Accumulate experiment rows incrementally; persist/resume from pickle |
| `read_typed_csv(path)` | Parse `n_vars; [constraint, ...]` CSV format |
| `remap_constraint_to_vars(c, vars)` | Embed zero-indexed constraint into QUBO variable positions |
