# slurm

Utilities for running batch jobs on a SLURM cluster.

## Files

| File | Purpose |
|---|---|
| `generate_params.py` | Convenience wrapper: delegates to `run/create_vcg_database.py` and `run/generate_experiment_params.py` to produce JSONL task lists |

## Usage

```bash
# Generate both VCG and experiment task lists:
python slurm/generate_params.py

# VCG task list only:
python slurm/generate_params.py --vcg

# Experiment task list only:
python slurm/generate_params.py --hybrid

# Custom output paths and task count:
python slurm/generate_params.py \
    --vcg-params run/params/vcg_params.jsonl \
    --experiment-params run/params/experiment_params.jsonl \
    --max-tasks 500
```

## Submitting array jobs

After generating parameter files, submit SLURM array jobs that call the
`--task-id N` mode of the relevant `run/` script:

```bash
# VCG training array (N = number of lines in vcg_params.jsonl):
sbatch --array=0-<N-1> slurm/vcg_array.sh

# Experiment array (N = number of lines in experiment_params.jsonl):
sbatch --array=0-<N-1> slurm/experiment_array.sh
```

Each task writes its result to a per-task pickle in a `pending/` directory.
After all tasks complete, merge results back into the main DB or output file:

```bash
# Merge VCG results:
python run/create_vcg_database.py --merge \
    --pending-dir gadgets/pending/ \
    --db gadgets/gadget_db.pkl

# Merge experiment results:
python run/run_hybrid_vs_penalty.py \
    --merge --pending-dir results/pending/ \
    --output results/hybrid_vs_penalty.pkl
```

See `run/README.md` for the full workflow.
