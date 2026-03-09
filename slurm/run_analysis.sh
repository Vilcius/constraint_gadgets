#!/bin/bash
#SBATCH -J run_analysis
#SBATCH -A SIP-UTK0040
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -p campus
#SBATCH -e job_files/%j.err
#SBATCH -o job_files/%j.out
#SBATCH --qos=campus

# Step 1: Split raw result pickles into purpose-specific DataFrames
#         (computes p_feasible, p_optimal, depth here — downstream scripts
#          never need raw counts or Hamiltonians).
# Step 2: Run plots and statistical tests on the split files.
#
# Called by submit_all.sh:
#   sbatch --dependency=afterok:<exp_merge_id> slurm/run_analysis.sh /abs/path/to/project

module load anaconda3/2021.05
source $ANACONDA_SH

if [[ -n "$1" ]]; then
    PROJECT_ROOT="$1"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$DIR/.." && pwd)"
fi
RESULTS_DIR="$PROJECT_ROOT/results"
OUTPUT_DIR="$PROJECT_ROOT/analysis_output"

echo "=== Step 1: Splitting results ==="
python3.11 "$PROJECT_ROOT/analyze_results/split_results.py" \
    --vcg-dir    "$PROJECT_ROOT/gadgets/pending/" \
    --hybrid     "$RESULTS_DIR/hybrid_vs_penalty.pkl" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=== Step 2: Running analysis and plots ==="
python3.11 "$PROJECT_ROOT/analyze_results/main_analysis.py" \
    --vcg-ar    "$RESULTS_DIR/vcg_ar.pkl" \
    --vcg-res   "$RESULTS_DIR/vcg_resources.pkl" \
    --comp-ar   "$RESULTS_DIR/comparison_ar.pkl" \
    --comp-res  "$RESULTS_DIR/comparison_resources.pkl" \
    --output-dir "$OUTPUT_DIR"
