#!/bin/bash
#SBATCH -J experiment_merge
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

python3.11 "$PROJECT_ROOT/run/run_hybrid_vs_penalty.py" \
    --merge \
    --pending-dir "$PROJECT_ROOT/results/pending/" \
    --output "$PROJECT_ROOT/results/hybrid_vs_penalty.pkl"
