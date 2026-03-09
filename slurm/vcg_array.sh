#!/bin/bash
#SBATCH -J vcg_training_array
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -p campus
#SBATCH -e job_files/%A_%a.err
#SBATCH -o job_files/%A_%a.out
#SBATCH --qos=campus

module load anaconda3/2021.05
source $ANACONDA_SH

# BASH_SOURCE[0] resolves to SLURM's spool copy, not the original script.
# Derive PROJECT_ROOT from $1 (always an absolute path from submit_all.sh).
# Fall back to BASH_SOURCE only for interactive use.
if [[ -n "$1" ]]; then
    PARAMS_FILE="$1"
    PROJECT_ROOT="$(cd "$(dirname "$PARAMS_FILE")/../.." && pwd)"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PARAMS_FILE="$DIR/../run/params/vcg_params.jsonl"
    PROJECT_ROOT="$(cd "$DIR/.." && pwd)"
fi

mkdir -p "$PROJECT_ROOT/gadgets/pending"

python3.11 "$PROJECT_ROOT/run/create_vcg_database.py" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --params-out "$PARAMS_FILE" \
    --db "$PROJECT_ROOT/gadgets/gadget_db.pkl" \
    --pending-dir "$PROJECT_ROOT/gadgets/pending/"
