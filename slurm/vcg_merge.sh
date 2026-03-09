#!/bin/bash
#SBATCH -J vcg_merge
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
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

python3.11 "$PROJECT_ROOT/run/create_vcg_database.py" \
    --merge \
    --pending-dir "$PROJECT_ROOT/gadgets/pending/" \
    --db "$PROJECT_ROOT/gadgets/gadget_db.pkl"
