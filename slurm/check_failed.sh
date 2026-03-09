#!/bin/bash
# Find task IDs that did not produce a result pickle, so they can be resubmitted.
#
# Usage:
#   bash slurm/check_failed.sh vcg   <N>   # check VCG tasks 0..(N-1)
#   bash slurm/check_failed.sh exp   <N>   # check experiment tasks 0..(N-1)
#
# Prints a comma-separated list of failed IDs suitable for --array=:
#   sbatch --array=<list> slurm/vcg_array.sh run/params/vcg_params.jsonl

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"

MODE="$1"
N="$2"

if [[ -z "$MODE" || -z "$N" ]]; then
    echo "Usage: $0 {vcg|exp} <N>"
    exit 1
fi

if [[ "$MODE" == "vcg" ]]; then
    PENDING_DIR="$PROJECT_ROOT/gadgets/pending"
elif [[ "$MODE" == "exp" ]]; then
    PENDING_DIR="$PROJECT_ROOT/results/pending"
else
    echo "Unknown mode '$MODE'. Use 'vcg' or 'exp'."
    exit 1
fi

failed=()
for i in $(seq 0 $((N - 1))); do
    if [[ ! -f "$PENDING_DIR/task_${i}.pkl" ]]; then
        failed+=("$i")
    fi
done

if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All $N tasks completed successfully."
else
    echo "Failed / missing task IDs (${#failed[@]} of $N):"
    joined=$(IFS=,; echo "${failed[*]}")
    echo "  $joined"
    echo ""
    if [[ "$MODE" == "vcg" ]]; then
        echo "Resubmit with:"
        echo "  sbatch --array=$joined slurm/vcg_array.sh $PROJECT_ROOT/run/params/vcg_params.jsonl"
        echo ""
        echo "Then re-merge:"
        echo "  python run/create_vcg_database.py --merge --pending-dir gadgets/pending/ --db gadgets/gadget_db.pkl"
    else
        echo "Resubmit with:"
        echo "  sbatch --array=$joined slurm/experiment_array.sh $PROJECT_ROOT/run/params/experiment_params.jsonl"
        echo ""
        echo "Then re-merge:"
        echo "  python run/run_hybrid_vs_penalty.py --merge --pending-dir results/pending/ --output results/hybrid_vs_penalty.pkl"
    fi
fi
