#!/bin/bash
# Submit rerun experiments in priority order.
# Priority 1 (error):   overlapping + disjoint jobs that failed with the
#                        control-wires bug — submit immediately, no dependency.
# Priority 2 (missing): experiments that never ran — start after error jobs.
# Priority 3 (oom):     memory-heavy overlapping experiments — start last.
#
# Usage (from project root):
#   bash slurm/submit_reruns.sh
#
# Output goes to the same pending dirs as the original runs so that existing
# results are preserved and the merge step picks everything up together.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"
RERUN="$PROJECT_ROOT/run/rerun"

mkdir -p "$DIR/job_files"

OV_PENDING="$PROJECT_ROOT/results/overlapping/pending"
DJ_PENDING="$PROJECT_ROOT/results/disjoint/pending"

# Helper: count lines in a jsonl file
n_lines() { wc -l < "$1" | tr -d ' '; }

# ── Priority 1: error experiments (bug-fix reruns) ───────────────────────────

OV_ERR_N=$(( $(n_lines "$RERUN/overlapping_error.jsonl") - 1 ))
ov_err_id=$(sbatch \
    --array=0-${OV_ERR_N} \
    "$DIR/rerun_array.sh" "$RERUN/overlapping_error.jsonl" "$OV_PENDING" \
    | awk '{print $4}')
echo "P1 overlapping error  ($(( OV_ERR_N + 1 )) experiments) — job $ov_err_id"

DJ_ERR_N=$(( $(n_lines "$RERUN/disjoint_error.jsonl") - 1 ))
dj_err_id=$(sbatch \
    --array=0-${DJ_ERR_N} \
    "$DIR/rerun_array.sh" "$RERUN/disjoint_error.jsonl" "$DJ_PENDING" \
    | awk '{print $4}')
echo "P1 disjoint error     ($(( DJ_ERR_N + 1 )) experiments) — job $dj_err_id"

# ── Priority 2: missing experiments ─────────────────────────────────────────

OV_MISS_N=$(( $(n_lines "$RERUN/overlapping_missing.jsonl") - 1 ))
ov_miss_id=$(sbatch \
    --dependency=afterany:${ov_err_id}:${dj_err_id} \
    --array=0-${OV_MISS_N} \
    "$DIR/rerun_array.sh" "$RERUN/overlapping_missing.jsonl" "$OV_PENDING" \
    | awk '{print $4}')
echo "P2 overlapping missing ($(( OV_MISS_N + 1 )) experiments) — job $ov_miss_id (after P1)"

DJ_MISS_N=$(( $(n_lines "$RERUN/disjoint_missing.jsonl") - 1 ))
dj_miss_id=$(sbatch \
    --dependency=afterany:${ov_err_id}:${dj_err_id} \
    --array=0-${DJ_MISS_N} \
    "$DIR/rerun_array.sh" "$RERUN/disjoint_missing.jsonl" "$DJ_PENDING" \
    | awk '{print $4}')
echo "P2 disjoint missing   ($(( DJ_MISS_N + 1 )) experiments) — job $dj_miss_id (after P1)"

# ── Priority 3: OOM experiments (overlapping only) ───────────────────────────

OV_OOM_N=$(( $(n_lines "$RERUN/overlapping_oom.jsonl") - 1 ))
ov_oom_id=$(sbatch \
    --dependency=afterany:${ov_miss_id}:${dj_miss_id} \
    --array=0-${OV_OOM_N} \
    "$DIR/rerun_array.sh" "$RERUN/overlapping_oom.jsonl" "$OV_PENDING" \
    | awk '{print $4}')
echo "P3 overlapping OOM    ($(( OV_OOM_N + 1 )) experiments) — job $ov_oom_id (after P2)"

echo ""
echo "All rerun jobs queued. Check status with: squeue -u \$USER"
echo ""
echo "Once complete, merge results with:"
echo "  python run/run_hybrid_vs_penalty.py --merge \\"
echo "      --pending-dir results/overlapping/pending/ \\"
echo "      --output results/overlapping/hybrid_vs_penalty.pkl"
echo "  python run/run_hybrid_vs_penalty.py --merge \\"
echo "      --pending-dir results/disjoint/pending/ \\"
echo "      --output results/disjoint/hybrid_vs_penalty.pkl"
