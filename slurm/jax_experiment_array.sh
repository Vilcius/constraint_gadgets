#!/bin/bash
#SBATCH -J jax_experiment_array
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

# Arguments: $1 = PARAMS_FILE  $2 = PENDING_DIR
if [[ -n "$1" ]]; then
    PARAMS_FILE="$1"
    PROJECT_ROOT="$(cd "$(dirname "$PARAMS_FILE")/../.." && pwd)"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PARAMS_FILE="$DIR/../run/params/experiment_params.jsonl"
    PROJECT_ROOT="$(cd "$DIR/.." && pwd)"
fi

PENDING_DIR="${2:-$PROJECT_ROOT/results/jax_pending_overlapping}"
mkdir -p "$PENDING_DIR"

python3.11 "$PROJECT_ROOT/run/run_hybrid_vs_penalty_jax.py" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --params  "$PARAMS_FILE" \
    --db      "$PROJECT_ROOT/gadgets/noflag_db.pkl" \
    --pending-dir "$PENDING_DIR"
