#!/bin/bash
#SBATCH -J jax_experiment_merge
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

# Arguments: $1 = PROJECT_ROOT  $2 = PENDING_DIR  $3 = OUTPUT_PKL
PROJECT_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PENDING_DIR="${2:-$PROJECT_ROOT/results/jax_pending_overlapping}"
OUTPUT="${3:-$PROJECT_ROOT/results/overlapping/hybrid_vs_penalty_jax.pkl}"

python3.11 "$PROJECT_ROOT/run/run_hybrid_vs_penalty_jax.py" \
    --merge \
    --pending-dir "$PENDING_DIR" \
    --output      "$OUTPUT"
