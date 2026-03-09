#!/bin/bash
#SBATCH -J vcg_launcher
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH -p campus
#SBATCH -e job_files/%j.err
#SBATCH -o job_files/%j.out
#SBATCH --qos=campus

# Launcher for the VCG training stage.
# Generates the VCG task list, submits the training array and merge job,
# then submits experiment_launcher.sh to run after the merge completes.
#
# Called by submit_all.sh:
#   sbatch slurm/vcg_launcher.sh /abs/path/to/project

module load anaconda3/2021.05
source $ANACONDA_SH

PROJECT_ROOT="$1"
DIR="$PROJECT_ROOT/slurm"
VCG_PARAMS="$PROJECT_ROOT/run/params/vcg_params.jsonl"

mkdir -p "$PROJECT_ROOT/gadgets/pending"

# ── 1. Generate VCG task list ─────────────────────────────────────────────────
python3.11 "$PROJECT_ROOT/run/create_vcg_database.py" \
    --generate-params \
    --params-out "$VCG_PARAMS" \
    --data-dir "$PROJECT_ROOT/data/"

N_VCG=$(wc -l < "$VCG_PARAMS")
echo "VCG tasks: $N_VCG"

# ── 2. Submit VCG training array ──────────────────────────────────────────────
vcg_array_id=$(sbatch --array=0-$((N_VCG - 1)) \
    "$DIR/vcg_array.sh" "$VCG_PARAMS" \
    | awk '{print $4}')
echo "VCG training array submitted: job $vcg_array_id ($N_VCG tasks)"

# ── 3. Submit VCG merge (after array finishes) ────────────────────────────────
vcg_merge_id=$(sbatch \
    --dependency=afterany:${vcg_array_id} \
    "$DIR/vcg_merge.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "VCG merge submitted: job $vcg_merge_id (after $vcg_array_id)"

# ── 4. Submit experiment launcher (after VCG merge succeeds) ──────────────────
exp_launcher_id=$(sbatch \
    --dependency=afterok:${vcg_merge_id} \
    "$DIR/experiment_launcher.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "Experiment launcher submitted: job $exp_launcher_id (after $vcg_merge_id)"
