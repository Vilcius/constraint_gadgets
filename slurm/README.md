# slurm

Utilities for running batch jobs on a SLURM cluster.

## Files

| File | Purpose |
|---|---|
| `generate_params.py` | Convenience wrapper: delegates to `run/create_vcg_database.py` and `run/generate_experiment_params.py` to produce JSONL task lists |
| `generate_vcg_params.sh` | Single-node job: runs `create_vcg_database.py --generate-params` to produce `run/params/vcg_params.jsonl` |
| `vcg_array.sh` | SLURM array script: trains one VCG per task (`--task-id $SLURM_ARRAY_TASK_ID`) |
| `vcg_merge.sh` | Single-node job: merges per-task VCG pickles into `gadgets/gadget_db.pkl` |
| `generate_experiment_params.sh` | Single-node job: runs `generate_experiment_params.py` to produce `run/params/experiment_params.jsonl` |
| `experiment_array.sh` | SLURM array script: runs one HybridQAOA + PenaltyQAOA experiment per task |
| `experiment_merge.sh` | Single-node job: merges per-task experiment pickles into `results/hybrid_vs_penalty.pkl` |
| `run_analysis.sh` | Single-node job: splits results and runs analysis/plots |
| `submit_all.sh` | Full pipeline: chains all 6 steps via SLURM dependencies |
| `check_failed.sh` | Utility: reports failed tasks from `results/pending/` and `gadgets/pending/` |

## Usage

### Full pipeline (recommended)

From the project root on the cluster:

```bash
bash slurm/submit_all.sh
```

This submits all 6 steps in dependency order and prints all job IDs.
Final results land in `results/hybrid_vs_penalty.pkl`.

### Step by step

```bash
# 1. Generate VCG task list
python run/create_vcg_database.py \
    --generate-params --params-out run/params/vcg_params.jsonl \
    --data-dir data/
N_VCG=$(wc -l < run/params/vcg_params.jsonl)

# 2. Submit VCG training array (0-indexed)
sbatch --array=0-$((N_VCG - 1)) slurm/vcg_array.sh run/params/vcg_params.jsonl

# 3. After all VCG tasks finish, merge
sbatch slurm/vcg_merge.sh $PWD   # or: python run/create_vcg_database.py --merge ...

# 4. Generate experiment task list
python run/generate_experiment_params.py \
    --output run/params/experiment_params.jsonl --max-tasks 500 --data-dir data/
N_EXP=$(wc -l < run/params/experiment_params.jsonl)

# 5. Submit experiment array
sbatch --array=0-$((N_EXP - 1)) slurm/experiment_array.sh run/params/experiment_params.jsonl

# 6. After all experiment tasks finish, merge
sbatch slurm/experiment_merge.sh $PWD   # or: python run/run_hybrid_vs_penalty.py --merge ...
```

See `run/README.md` for the full workflow description.
