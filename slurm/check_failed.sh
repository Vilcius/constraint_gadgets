#!/bin/bash
# Find experiment task IDs that did not produce a result pickle, and show
# error messages from any logged .failed.json files so they can be diagnosed
# and resubmitted.
#
# Usage:
#   bash slurm/check_failed.sh overlapping <N>   # check overlapping tasks 0..(N-1)
#   bash slurm/check_failed.sh disjoint    <N>   # check disjoint tasks 0..(N-1)
#
# Prints a comma-separated list of failed IDs suitable for --array=:
#   sbatch --array=<list> slurm/experiment_array.sh \
#       run/params/experiment_params_<split>.jsonl results/pending_<split>/

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"

SPLIT="$1"
N="$2"

if [[ -z "$SPLIT" || -z "$N" ]]; then
    echo "Usage: $0 {overlapping|disjoint} <N>"
    exit 1
fi

if [[ "$SPLIT" == "overlapping" || "$SPLIT" == "disjoint" ]]; then
    PENDING_DIR="$PROJECT_ROOT/results/pending_${SPLIT}"
    PARAMS_FILE="run/params/experiment_params_${SPLIT}.jsonl"
    OUTPUT_PKL="results/${SPLIT}/pc_qaoa_vs_penalty.pkl"
    MERGE_CMD="python run/run_pc_qaoa_vs_penalty.py --merge --pending-dir results/pending_${SPLIT}/ --output $OUTPUT_PKL"
else
    echo "Unknown split '$SPLIT'. Use 'overlapping' or 'disjoint'."
    exit 1
fi

failed=()
for i in $(seq 0 $((N - 1))); do
    if [[ ! -f "$PENDING_DIR/cop_${i}.pkl" ]]; then
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
    log="$PENDING_DIR/cop_${i}.failed.json"
    if [[ -f "$log" ]]; then
        if [[ $has_logs -eq 0 ]]; then
            echo "Error details (from .failed.json logs):"
            has_logs=1
        fi
        echo "  --- cop $i ---"
        python3 -c "
import json, sys
with open('$log') as f:
    d = json.loads(f.read())
print('  timestamp:', d.get('timestamp','?'))
print('  error    :', d.get('error','?'))
cop = d.get('cop', {})
print('  cop      :', json.dumps(cop)[:120])
"
    fi
done
[[ $has_logs -eq 0 ]] && echo "(No .failed.json logs found — tasks likely hit the time limit or were cancelled.)"

echo ""
echo "Resubmit with:"
echo "  sbatch --array=$joined slurm/experiment_array.sh \\"
echo "      $PROJECT_ROOT/$PARAMS_FILE \\"
echo "      $PENDING_DIR"
echo ""
echo "Then re-merge:"
echo "  $MERGE_CMD"
