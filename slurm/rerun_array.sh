#!/bin/bash
# Generic rerun array job.
# Usage (via submit_reruns.sh — do not submit directly):
#   sbatch --array=0-N rerun_array.sh <params_file> <pending_dir>
#
# $1 — absolute path to a rerun .jsonl params file (run/rerun/*.jsonl)
# $2 — absolute path to the pending output directory
#SBATCH -J rerun_array
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=168:00:00
#SBATCH -p campus
#SBATCH -e job_files/%A_%a.err
#SBATCH -o job_files/%A_%a.out
#SBATCH --qos=campus

module load anaconda3/2021.05
source $ANACONDA_SH

PARAMS_FILE="$1"
PENDING_DIR="$2"
PROJECT_ROOT="$(cd "$(dirname "$PARAMS_FILE")/../.." && pwd)"

mkdir -p "$PENDING_DIR"

python3.11 "$PROJECT_ROOT/run/run_hybrid_vs_penalty.py" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --params  "$PARAMS_FILE" \
    --db      "$PROJECT_ROOT/gadgets/noflag_db.pkl" \
    --pending-dir "$PENDING_DIR"
