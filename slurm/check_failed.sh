#!/bin/bash
# Find task IDs that did not produce a result pickle, and show error messages
# from any logged .failed.json files, so they can be diagnosed and resubmitted.
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
    ARRAY_SCRIPT="slurm/vcg_array.sh"
    PARAMS_FILE="run/params/vcg_params.jsonl"
    MERGE_CMD="python run/create_vcg_database.py --merge --pending-dir gadgets/pending/ --db gadgets/gadget_db.pkl"
elif [[ "$MODE" == "exp" ]]; then
    PENDING_DIR="$PROJECT_ROOT/results/pending"
    ARRAY_SCRIPT="slurm/experiment_array.sh"
    PARAMS_FILE="run/params/experiment_params_overlapping.jsonl"
    MERGE_CMD="python run/run_hybrid_vs_penalty.py --merge --pending-dir results/pending/ --output results/hybrid_vs_penalty.pkl"
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
    exit 0
fi

echo "Failed / missing task IDs (${#failed[@]} of $N):"
joined=$(IFS=,; echo "${failed[*]}")
echo "  $joined"
echo ""

# Show error messages from any logged .failed.json files
has_logs=0
for i in "${failed[@]}"; do
    log="$PENDING_DIR/task_${i}.failed.json"
    if [[ -f "$log" ]]; then
        if [[ $has_logs -eq 0 ]]; then
            echo "Error details (from .failed.json logs):"
            has_logs=1
        fi
        echo "  --- task $i ---"
        # Print timestamp and first line of error (avoid printing full traceback)
        python3 -c "
import json, sys
with open('$log') as f:
    d = json.loads(f.read())
print('  timestamp:', d.get('timestamp','?'))
print('  error    :', d.get('error','?'))
task = d.get('task', {})
print('  task     :', json.dumps(task)[:120])
"
    fi
done
[[ $has_logs -eq 0 ]] && echo "(No .failed.json logs found — tasks likely hit the time limit or were cancelled.)"

echo ""
echo "Resubmit with:"
echo "  sbatch --array=$joined $ARRAY_SCRIPT $PROJECT_ROOT/$PARAMS_FILE"
echo ""
echo "Then re-merge:"
echo "  $MERGE_CMD"
