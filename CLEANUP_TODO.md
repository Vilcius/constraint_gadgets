# Cleanup & Fix Pipeline — Task List

Status legend: [ ] pending  [x] done  [~] in progress

---

## Phase 1 — Remove Clutter

- [x] **#1** Remove stale result files and old DBs
- [x] **#2** Remove dead `run/` scripts
- [x] **#3** Remove dead `slurm/` scripts
- [x] **#4** Remove `progress/` directory
- [x] **#5** Remove dead `analyze_results/` scripts
  - `analyze_results/data_loader.py` — deleted
  - `analyze_results/results_helper.py` — removed `GadgetDatabase`, `collect_vcg_data`, `collect_hybrid_data`, `collect_penalty_data`, `remap_to_zero_indexed`

---

## Phase 2 — Fix Pipeline

- [ ] **#6** Fix `generate_experiment_params.py`
  - After generating tasks, extract all unique VCG-type constraints
    (families: knapsack, quadratic_knapsack)
  - Write to `run/params/vcg_params_experiments.jsonl`
  - This file feeds `create_vcg_database.py`

- [x] **#7** Fix `slurm/experiment_array.sh` DB path
  - Changed `--db gadgets/noflag_db.pkl` → `--db gadgets/vcg_db.pkl`

- [x] **#8** Fix `slurm/submit.sh` job ordering
  - Created `slurm/vcg_train.sh` (single job, 8 workers, replaces stale `vcg_array.sh`)
  - Both experiment arrays now use `--dependency=afterok:<vcg_job_id>`

---

## Phase 3 — Rerun Everything

- [ ] **#9** Regenerate param files
  - `python run/generate_vcg_params.py` → `run/params/vcg_params_experiments.jsonl`
  - `python run/generate_experiment_params.py` (overlapping + disjoint)

- [ ] **#10** Train VCG database from experiment constraints
  - `python run/create_vcg_database.py --params run/params/vcg_params_experiments.jsonl --db gadgets/vcg_db.pkl`

- [ ] **#11** Submit all 1000 experiments via SLURM
  - `slurm/submit.sh` on server
  - Monitor for failures

- [ ] **#12** Merge experiment results
  - Run merge scripts for overlapping + disjoint
  - Verify 500 rows each

---

## Phase 4 — Post-Processing & Analysis

- [ ] **#13** Run `split_results.py` + `compute_circuit_resources.py`
  - Produces `comparison_ar.pkl` and `circuit_resources.pkl` for all 1000 experiments

- [ ] **#14** Run `main_analysis.py`
  - Regenerates all figures, tables, and statistical tests
