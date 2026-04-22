#!/bin/bash
#SBATCH -J vcg_train
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -p campus
#SBATCH -e job_files/%j.err
#SBATCH -o job_files/%j.out
#SBATCH --qos=campus

module load anaconda3/2021.05
source $ANACONDA_SH

if [[ -n "$1" ]]; then
    PROJECT_ROOT="$1"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$DIR/.." && pwd)"
fi

PARAMS="$PROJECT_ROOT/run/params/vcg_params_experiments.jsonl"
DB="$PROJECT_ROOT/gadgets/vcg_db.pkl"

mkdir -p "$PROJECT_ROOT/gadgets"

python3.11 "$PROJECT_ROOT/run/create_vcg_database.py" \
    --params  "$PARAMS" \
    --db      "$DB" \
    --workers 8
