#!/bin/bash
#SBATCH -J experiment_launcher
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH -p campus
#SBATCH -e job_files/%j.err
#SBATCH -o job_files/%j.out
#SBATCH --qos=campus

# Launcher for the experiment stage.
# Generates the experiment task list, submits the experiment array and merge job.
# Submitted automatically by vcg_launcher.sh after the VCG merge succeeds.
#
# Can also be run standalone after the gadget DB is ready:
#   sbatch slurm/experiment_launcher.sh /abs/path/to/project

module load anaconda3/2021.05
source $ANACONDA_SH

PROJECT_ROOT="$1"
DIR="$PROJECT_ROOT/slurm"
EXP_PARAMS="$PROJECT_ROOT/run/params/experiment_params.jsonl"

mkdir -p "$PROJECT_ROOT/results/pending"

# ── 1. Generate experiment task list ──────────────────────────────────────────
python3.11 "$PROJECT_ROOT/run/generate_experiment_params.py" \
    --output "$EXP_PARAMS" \
    --max-tasks 500 \
    --data-dir "$PROJECT_ROOT/data/"

N_EXP=$(wc -l < "$EXP_PARAMS")
echo "Experiment tasks: $N_EXP"

# ── 2. Submit experiment array ────────────────────────────────────────────────
exp_array_id=$(sbatch --array=0-$((N_EXP - 1)) \
    "$DIR/experiment_array.sh" "$EXP_PARAMS" \
    | awk '{print $4}')
echo "Experiment array submitted: job $exp_array_id ($N_EXP tasks)"

# ── 3. Submit experiment merge (after array finishes) ─────────────────────────
exp_merge_id=$(sbatch \
    --dependency=afterany:${exp_array_id} \
    "$DIR/experiment_merge.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "Experiment merge submitted: job $exp_merge_id (after $exp_array_id)"

echo ""
echo "Final results will be at: $PROJECT_ROOT/results/hybrid_vs_penalty.pkl"
