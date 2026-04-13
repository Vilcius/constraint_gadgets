#!/bin/bash
# Submit all 1000 JAX experiments (500 overlapping + 500 disjoint) via SLURM.
#
# Usage (from project root):
#   bash slurm/submit_jax.sh
#
# Each set submits an array + a dependent merge job.
# Merge also builds problem_table_jax.pkl automatically.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"

OVL_PARAMS="$PROJECT_ROOT/run/params/experiment_params.jsonl"
DIS_PARAMS="$PROJECT_ROOT/run/params/experiment_params_disjoint.jsonl"

OVL_PENDING="$PROJECT_ROOT/results/jax_pending_overlapping"
DIS_PENDING="$PROJECT_ROOT/results/jax_pending_disjoint"

OVL_OUTPUT="$PROJECT_ROOT/results/overlapping/hybrid_vs_penalty_jax.pkl"
DIS_OUTPUT="$PROJECT_ROOT/results/disjoint/hybrid_vs_penalty_jax.pkl"

mkdir -p "$DIR/job_files"
mkdir -p "$OVL_PENDING"
mkdir -p "$DIS_PENDING"
mkdir -p "$PROJECT_ROOT/results/overlapping"
mkdir -p "$PROJECT_ROOT/results/disjoint"

# ── Overlapping: 500 tasks, array 0-499 ──────────────────────────────────────
ovl_array_id=$(sbatch \
    --array=0-499 \
    "$DIR/jax_experiment_array.sh" "$OVL_PARAMS" "$OVL_PENDING" \
    | awk '{print $4}')
echo "Overlapping array submitted — job $ovl_array_id (tasks 0-499)"

ovl_merge_id=$(sbatch \
    --dependency=afterany:${ovl_array_id} \
    "$DIR/jax_experiment_merge.sh" "$PROJECT_ROOT" "$OVL_PENDING" "$OVL_OUTPUT" \
    | awk '{print $4}')
echo "Overlapping merge submitted — job $ovl_merge_id (after $ovl_array_id)"

# ── Disjoint: 500 tasks, array 0-499 ─────────────────────────────────────────
dis_array_id=$(sbatch \
    --array=0-499 \
    "$DIR/jax_experiment_array.sh" "$DIS_PARAMS" "$DIS_PENDING" \
    | awk '{print $4}')
echo "Disjoint array submitted — job $dis_array_id (tasks 0-499)"

dis_merge_id=$(sbatch \
    --dependency=afterany:${dis_array_id} \
    "$DIR/jax_experiment_merge.sh" "$PROJECT_ROOT" "$DIS_PENDING" "$DIS_OUTPUT" \
    | awk '{print $4}')
echo "Disjoint merge submitted — job $dis_merge_id (after $dis_array_id)"

echo ""
echo "All jobs queued. Monitor with: squeue -u \$USER"
echo "Outputs:"
echo "  $OVL_OUTPUT  (+problem_table_jax.csv/pkl)"
echo "  $DIS_OUTPUT  (+problem_table_jax.csv/pkl)"
