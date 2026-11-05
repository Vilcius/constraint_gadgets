# Uncapped Gate-Budget Comparison

Compare QAOA+ at **one layer** against deep StatePrep PC-QAOA using approximately the same total NISQ gate count. There is no depth cap. The fixed-depth one/five comparison remains in the original results notebook; it is not another target in this experiment.

For each COP, select the largest whole PC-QAOA depth that fits the QAOA+ one-layer gate budget, with a minimum of one. Whole layers generally prevent exact equality. All 500 COPs remain eligible, including minimum-one-layer exceptions that exceed the budget. The notebook reports these exceptions and shows actual gate counts.

## Settings and Reuse

The dataset, objectives, constraints, partition, squared-residual slack encoding, penalties, exact gadgets, mixers and angle convention are unchanged. New training uses uncompiled NumPy statevector evolution with exact adjoint gradients, standard QAOA parameters, ten restarts, fifty Adam steps per depth, learning rate 0.01 and 10,000 final shots. NumPy PCG64 uses the original integer seed derivation but a different random sequence from JAX.

| Variant | COPs | Targets Already at Depths 1–5 | Require Deeper Training | Maximum Target Depth | Minimum-One-Layer Exceptions |
|---|---:|---:|---:|---:|---:|
| StatePrep | 500 | 233 | 267 | 433 | 133 |
| VCG, Optional | 500 | 329 | 171 | 90 | 181 |

StatePrep requires 18,483 additional depth optimizations beyond the original five-depth runs, before accounting for any completed gate-budget training. Existing results are never replaced with a shallower result when the target has not finished.

The notebook automatically reads original saved target depths and completed gate-budget comparisons. The runner imports original depths and also reuses the longest committed prefix from the earlier output at `results/gate_budget/pcqaoa_stateprep/` (or `pcqaoa/`). Earlier two-reference outputs are retained as reuse sources, but their depth-five QAOA+ comparisons are not plotted or scheduled by the corrected runner. Imported settings, state probabilities and losses are verified against the current circuit.

A machine needs these existing inputs:

- `final_experiment/random_cops/artifacts/manifest.json`
- `results/resource_estimates/final_circuit_resources.csv`
- `results/stateprep_resource_estimates/stateprep_circuit_resources.csv`
- `results/pcqaoa_stateprep/`, including its run metadata and per-COP depths
- `results/pcqaoa/` for optional regular VCG PC-QAOA
- `results/qaoa_plus/` for saved one-layer QAOA+ results
- `tuning/results/full/selected_vcg_db.pkl` relative to the repository root

Other `results/` paths are relative to `final_experiment/random_cops/`. Use the existing dataset.

## Run Commands

From `~/PCQAOA/code-tuning`:

```bash
conda activate testWH
python final_experiment/random_cops/run_gate_budget_stateprep.py prepare
python final_experiment/random_cops/run_gate_budget_stateprep.py smoke --workers 2 --threads 6
python final_experiment/random_cops/run_gate_budget_stateprep.py run --workers 2 --threads 6
python final_experiment/random_cops/run_gate_budget_stateprep.py resume --workers 2 --threads 6
python final_experiment/random_cops/run_gate_budget_stateprep.py status
```

Full outputs go to `results/gate_budget/pcqaoa_stateprep_qaoa_p1/`. The corrected default schedules only QAOA+ one-layer budgets. If a previous two-reference job is still running, stop it with Ctrl-C before starting the corrected command. Its saved depths remain available for reuse. Smoke runs use a separate directory with reduced budgets and are not plotted. Setup and notebook execution do not start full training.

Regular VCG PC-QAOA uses the same commands with `run_gate_budget_pcqaoa.py` and writes to `pcqaoa_qaoa_p1/`.

To divide work across two machines:

```bash
# Computer 1
python final_experiment/random_cops/run_gate_budget_stateprep.py run --workers 2 --threads 6 --shard-count 2 --shard-index 0 --output final_experiment/random_cops/results/gate_budget/pcqaoa_stateprep_shard_0

# Computer 2
python final_experiment/random_cops/run_gate_budget_stateprep.py run --workers 2 --threads 6 --shard-count 2 --shard-index 1 --output final_experiment/random_cops/results/gate_budget/pcqaoa_stateprep_shard_1
```

Combine both complete shard directories under one results root for plotting. Use fresh shard directories if those names already contain the earlier two-reference run. `--reuse-run PATH` can be repeated to import completed depths from additional old shard directories. Original saved depths remain the fallback for COPs those directories have not reached. Do not run duplicate full and sharded jobs for the same COPs.

## Checkpoints and Notebook

Checkpoints are atomic with a valid predecessor. Every ten optimizer steps, the active optimizer state and scalar progress are saved. Completed depth records are written once under `COP/depths/`; interrupted runs continue without repeating committed depths. Each task writes `comparison_qaoa_p1.json` and `result.json` when its target is reached. Incompatible source, settings, dataset or environment changes are rejected on resume. `export` writes `comparisons.csv`.

Open `notebooks/gate_budget_results.ipynb`. It prints expected, available, reused, running, failed and remaining counts. Saved original target results appear immediately, even before the new runner is launched. The notebook includes required-depth distributions; paired P(opt), P(feas) and AR_feas scatter plots; paired differences and performance by problem size; overlap results; and gate/qubit comparisons. Each pair uses the same COPs and its own target depth. It reads results only and never starts training or resource estimation.

All 500 COPs remain eligible by default. Set `INCLUDE_BUDGET_OVERRUNS=False` only for an explicitly restricted analysis. This removes 133 StatePrep or 181 VCG minimum-depth exceptions; it is not the default experiment. No observations are excluded based on performance. Intervals are pointwise 95% Student-t intervals across COPs, with restarts and shots excluded from the instance count.

Performance-by-size plots show QAOA+ at one layer and both PC-QAOA variants together. All three curves, and the corresponding paired-difference curves, use the same completed COPs; the notebook prints counts by size. Each PC-QAOA variant still uses its own gate-budget target depth. Separate scatter plots retain each variant's full available pair set.

Overlap plots combine disjoint supports and overlap levels one and two on the same axes, using method colors and separate gray legend entries for line/marker styles. Each PC-QAOA variant keeps its own paired COP set. A cumulative plot shows the distribution of paired P(opt) differences, and one stacked figure summarizes wins, ties and losses for all three metrics. Original paired scatter plots are retained. Replaced overlap and win-fraction exports are removed only after their consolidated replacements are saved.

Resource-by-size figures now combine all three methods on their common completed COPs. The gate-matching scatter also combines both PC-QAOA variants against one QAOA+ diagonal. Replaced resource exports are removed after the consolidated figures are saved.

The total optimization-time scatter uses the main notebook's method columns around simulated-qubit ticks. It reads newly measured uncompiled prefix timings from `results/uncompiled/`, replacing historical compiled prefix timings, and adds the already-recorded uncompiled extension time. QAOA+ uses its new depth-one timing. Only COPs with all required target results and timings are shown. These are reconstructed workload totals from separate measurements, not newly executed deep trajectories; machine/load and optimized-state differences remain possible. No compilation or compiled execution time is included.

Feasibility tables compare each target against the same COP at original depth five, grouped by target depth and support overlap.

Figures retain Computer Modern, method colors and separate PDF/PNG exports under `results/gate_budget/figures/`. Set `PCQAOA_GATE_BUDGET_RESULTS` to read another result root.

```bash
python final_experiment/random_cops/gate_budget/build_notebook.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s final_experiment/random_cops/tests -p test_gate_budget.py -v
```
