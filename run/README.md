# run

Scripts for building the VCG database and running PC-QAOA vs PenaltyQAOA experiments.

## Scripts

| File | Purpose |
|---|---|
| `generate_vcg_params.py` | Extract unique VCG constraints from experiment params → `vcg_params_experiments.jsonl` |
| `create_vcg_database.py` | Train all VCG gadgets (knapsack + quadratic-knapsack); save to `gadgets/vcg_db.pkl` |
| `generate_experiment_params.py` | Enumerate PC-QAOA vs PenaltyQAOA tasks; write overlapping and disjoint JSONL files |
| `run_pc_qaoa_vs_penalty.py` | Run one COP (SLURM) or merge pending results into a single DataFrame |

## Workflow

```
1. generate_vcg_params.py       → run/params/vcg_params_experiments.jsonl
2. create_vcg_database.py       → gadgets/vcg_db.pkl
3. generate_experiment_params.py → run/params/experiment_params_{overlapping,disjoint}.jsonl
4. run_pc_qaoa_vs_penalty.py    → results/{overlapping,disjoint}/pc_qaoa_vs_penalty.pkl
```

## generate_vcg_params.py

Reads the experiment param files and extracts all unique knapsack and
quadratic-knapsack constraints that need a trained VCG gadget.
Cardinality, flow, assignment, and independent-set constraints are handled
by exact state preparations and do not need VCGs.

```bash
python run/generate_vcg_params.py
# → run/params/vcg_params_experiments.jsonl
```

## create_vcg_database.py

Trains a VCG gadget for each constraint in the params file.
Skips constraints already present in the DB.

```bash
# Sequential (single machine):
python run/create_vcg_database.py \
    --params run/params/vcg_params_experiments.jsonl \
    --db gadgets/vcg_db.pkl

# Parallel (8 workers, used by slurm/vcg_train.sh):
python run/create_vcg_database.py \
    --params run/params/vcg_params_experiments.jsonl \
    --db gadgets/vcg_db.pkl \
    --workers 8

# Force retrain even if already in DB:
python run/create_vcg_database.py \
    --params run/params/vcg_params_experiments.jsonl \
    --db gadgets/vcg_db.pkl \
    --force
```

## generate_experiment_params.py

Enumerates 2–3 constraint COPs drawn from any supported family.
Generates both overlapping (variables may be shared across constraints)
and disjoint (all constraint variable sets are disjoint) splits.

```bash
# Overlapping (default):
python run/generate_experiment_params.py \
    --output run/params/experiment_params_overlapping.jsonl \
    --max-cops 250 --seed 42

# Disjoint:
python run/generate_experiment_params.py \
    --output run/params/experiment_params_disjoint.jsonl \
    --max-cops 250 --seed 42 --disjoint
```

Each line specifies `constraints`, `families`, `n_x`, and `qubo_idx`.

## run_pc_qaoa_vs_penalty.py

Runs PC-QAOA and PenaltyQAOA layer sweeps for a single COP or merges all
pending results. Both solvers use ma-QAOA angles with warm-started layer
growth. Supports resuming interrupted runs.

```bash
# Single COP (used by slurm/experiment_array.sh):
python run/run_pc_qaoa_vs_penalty.py \
    --params run/params/experiment_params_overlapping.jsonl \
    --cop-id 42 \
    --db gadgets/vcg_db.pkl \
    --pending-dir results/pending_overlapping/

# Merge SLURM results (used by slurm/experiment_merge.sh):
python run/run_pc_qaoa_vs_penalty.py \
    --merge \
    --pending-dir results/pending_overlapping/ \
    --output results/overlapping/pc_qaoa_vs_penalty.pkl

# Sequential (local, all COPs):
python run/run_pc_qaoa_vs_penalty.py \
    --params run/params/experiment_params_overlapping.jsonl \
    --db gadgets/vcg_db.pkl
```
