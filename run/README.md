# run

Scripts for building the VCG database and running HybridQAOA vs PenaltyQAOA experiments.
All scripts are runnable directly or importable as a library.

## Scripts

| File | Purpose |
|---|---|
| `add_to_vcg_database.py` | Train a single VCG (QAOA warm-start → ma-QAOA sweep) and add it to the gadget DB |
| `create_vcg_database.py` | Populate the full gadget DB for all knapsack and quadratic-knapsack constraints |
| `generate_experiment_params.py` | Enumerate HybridQAOA vs PenaltyQAOA experiment tasks and write a JSONL parameter file |
| `run_hybrid_vs_penalty.py` | Run the experiment sweep: HybridQAOA and PenaltyQAOA layer sweeps for each task |

## Workflow

```
1. create_vcg_database.py        ← train VCGs; saves gadgets/gadget_db.pkl
2. generate_experiment_params.py ← enumerate tasks; saves run/params/experiment_params.jsonl
3. run_hybrid_vs_penalty.py      ← run experiments; saves results/hybrid_vs_penalty.pkl
```

## add_to_vcg_database.py

Trains a VCG for one set of constraints and registers it in the gadget database.

**Training procedure:**
1. Single QAOA p=1 run (2 parameters, ~8 s) to obtain warm-start angles.
2. ma-QAOA layer sweep: p=1 seeded from QAOA angles; p>1 jointly re-optimises
   all layers warm-started from the previous depth. Stops when AR ≥ threshold.

```bash
python run/add_to_vcg_database.py \
    --constraints "5*x_0 + 10*x_1 + 1*x_2 <= 9" \
    --db gadgets/gadget_db.pkl

# With explicit budget:
python run/add_to_vcg_database.py \
    --constraints "5*x_0 + 10*x_1 + 1*x_2 <= 9" \
    --db gadgets/gadget_db.pkl \
    --ar-threshold 0.999 --max-layers 8 \
    --qaoa-restarts 5 --qaoa-steps 150 \
    --ma-restarts 20 --ma-steps 200
```

Library usage:
```python
from run.add_to_vcg_database import train_and_add
ar = train_and_add(
    constraints=["5*x_0 + 10*x_1 + 1*x_2 <= 9"],
    db_path="gadgets/gadget_db.pkl",
)
```

## create_vcg_database.py

Discovers all knapsack and quadratic-knapsack constraints (n ≥ 3) and trains
a VCG for each one. Skips constraints already present in the DB.

```bash
# Sequential (all constraints, ~8 hours):
python run/create_vcg_database.py

# SLURM parallel:
# Step 1 – write task list
python run/create_vcg_database.py --generate-params \
    --params-out run/params/vcg_params.jsonl

# Step 2 – submit array job (N = number of lines in vcg_params.jsonl)
#   sbatch --array=0-<N-1> slurm/vcg_array.sh

# Step 3 – merge per-task results
python run/create_vcg_database.py --merge \
    --pending-dir gadgets/pending/ \
    --db gadgets/gadget_db.pkl
```

## generate_experiment_params.py

Enumerates experiment combinations (1–2 structural + 1–2 penalty constraints)
and writes one JSON object per line to a JSONL file.

```bash
python run/generate_experiment_params.py \
    --output run/params/experiment_params.jsonl \
    --max-tasks 500 --seed 42
```

Each line specifies `structural_constraints`, `penalty_constraints`,
`structural_indices`, `penalty_indices`, `n_x`, and `qubo_idx`.

## run_hybrid_vs_penalty.py

Runs HybridQAOA and PenaltyQAOA layer sweeps for every experiment task.
Both solvers use ma-QAOA angles and warm-started layer growth.
Stops each solver when P(feasible) ≥ 0.75 or max layers is reached.

```bash
# Sequential:
python run/run_hybrid_vs_penalty.py \
    --params run/params/experiment_params.jsonl \
    --db gadgets/gadget_db.pkl

# Single SLURM task:
python run/run_hybrid_vs_penalty.py \
    --params run/params/experiment_params.jsonl \
    --task-id 42 \
    --db gadgets/gadget_db.pkl \
    --pending-dir results/pending/

# Merge results:
python run/run_hybrid_vs_penalty.py \
    --merge --pending-dir results/pending/ \
    --output results/hybrid_vs_penalty.pkl
```
