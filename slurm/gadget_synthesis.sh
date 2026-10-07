#!/bin/bash
#SBATCH -J gadget_synthesis
#SBATCH -A acf-utk0011
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -p campus
#SBATCH --qos=campus
#SBATCH --array=0-1
#SBATCH -e job_files/%x_%A_%a.err
#SBATCH -o job_files/%x_%A_%a.out

# Compare trained ma-QAOA VCGs with exact state-preparation synthesis.
# Array task 0: gadgets behind the submitted paper (support 3-5).
# Array task 1: Study A fidelity-loss gadgets (support 4-8).
# Submit from the repository root:  sbatch slurm/gadget_synthesis.sh

DBS=(
    "gadgets/vcg_db.pkl:paper_v2"
    "tuning/results/full/selected_vcg_db.pkl:study_a"
)
DB="${DBS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${DB##*:}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY="$HOME/.conda/envs/pcqaoa/bin/python"

cd "$SLURM_SUBMIT_DIR"
"$PY" analyze_results/compare_gadget_synthesis.py \
    --db "$DB" \
    --output "results/gadget_synthesis/$LABEL"
