# slurm

SLURM scripts for running the full experiment pipeline on an HPC cluster.

## Files

| File | Purpose |
|---|---|
| `submit.sh` | Full pipeline: submit VCG training + 500 experiments in dependency order |
| `vcg_train.sh` | Single job (8 workers): train all VCG gadgets → `gadgets/vcg_db.pkl` |
| `experiment_array.sh` | SLURM array: run one COP per task → `results/pending_*/cop_N.pkl` |
| `experiment_merge.sh` | Single job: merge pending results → `results/*/pc_qaoa_vs_penalty.pkl` |
| `check_failed.sh` | Utility: report failed tasks and print resubmit commands |

## Usage

### Full pipeline (recommended)

From the project root on the cluster:

```bash
bash slurm/submit.sh
```

This submits three stages in dependency order:

1. **VCG training** — single job, 8 parallel workers, trains all gadgets
2. **Experiment arrays** — 250 overlapping + 250 disjoint COPs, each as a SLURM array (depends on step 1)
3. **Merge jobs** — one per split, collects `cop_*.pkl` files into a single DataFrame (depends on step 2)

### Step by step

```bash
# 1. Train all VCG gadgets
sbatch slurm/vcg_train.sh $PWD

# 2. Submit experiment arrays (after VCG training completes)
sbatch --array=0-249 slurm/experiment_array.sh \
    $PWD/run/params/experiment_params_overlapping.jsonl \
    $PWD/results/pending_overlapping

sbatch --array=0-249 slurm/experiment_array.sh \
    $PWD/run/params/experiment_params_disjoint.jsonl \
    $PWD/results/pending_disjoint

# 3. Merge results (after each array completes)
sbatch slurm/experiment_merge.sh $PWD \
    results/pending_overlapping \
    results/overlapping/pc_qaoa_vs_penalty.pkl

sbatch slurm/experiment_merge.sh $PWD \
    results/pending_disjoint \
    results/disjoint/pc_qaoa_vs_penalty.pkl
```

### Post-processing (run after merge jobs complete)

The SLURM pipeline ends after the merge step. The following must be run manually (or via a new job):

```bash
# Split raw results into typed DataFrames (once per split)
python analyze_results/split_results.py \
    --pc-qaoa results/overlapping/pc_qaoa_vs_penalty.pkl \
    --output-dir results/overlapping/

python analyze_results/split_results.py \
    --pc-qaoa results/disjoint/pc_qaoa_vs_penalty.pkl \
    --output-dir results/disjoint/

# Compute circuit resources
python analyze_results/compute_circuit_resources.py
python analyze_results/compute_vcg_resources.py

# Combine splits, generate all plots, copy to paper/
python analyze_results/generate_plots.py
```

### Checking for failures

```bash
# Check which COPs failed and get resubmit commands
bash slurm/check_failed.sh overlapping 250
bash slurm/check_failed.sh disjoint 250
```

Failed tasks produce `cop_N.failed.json` logs in the pending directory with a timestamp and error message.
