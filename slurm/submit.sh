#!/bin/bash
# Submit VCG training + 500 COPs (250 overlapping + 250 disjoint) via SLURM.
#
# Job order:
#   1. vcg_train          — train all VCG gadgets (must finish first)
#   2. experiment arrays  — 250 overlapping + 250 disjoint (depend on vcg_train)
#   3. merge jobs         — one per set (depend on each array)
#
# Usage (from project root):
#   bash slurm/submit.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"

OVL_PARAMS="$PROJECT_ROOT/run/params/experiment_params_overlapping.jsonl"
DIS_PARAMS="$PROJECT_ROOT/run/params/experiment_params_disjoint.jsonl"

OVL_PENDING="$PROJECT_ROOT/results/pending_overlapping"
DIS_PENDING="$PROJECT_ROOT/results/pending_disjoint"

OVL_OUTPUT="$PROJECT_ROOT/results/overlapping/hybrid_vs_penalty.pkl"
DIS_OUTPUT="$PROJECT_ROOT/results/disjoint/hybrid_vs_penalty.pkl"

mkdir -p "$DIR/job_files"
mkdir -p "$OVL_PENDING"
mkdir -p "$DIS_PENDING"
mkdir -p "$PROJECT_ROOT/results/overlapping"
mkdir -p "$PROJECT_ROOT/results/disjoint"

# ── 1. VCG training (must complete before experiments) ───────────────────────
vcg_id=$(sbatch \
    "$DIR/vcg_train.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "VCG training submitted — job $vcg_id"

# ── 2. Overlapping: 250 COPs, array 0-249 ────────────────────────────────────
ovl_array_id=$(sbatch \
    --dependency=afterok:${vcg_id} \
    --array=0-249 \
    "$DIR/experiment_array.sh" "$OVL_PARAMS" "$OVL_PENDING" \
    | awk '{print $4}')
echo "Overlapping array submitted — job $ovl_array_id (COPs 0-249, after $vcg_id)"

ovl_merge_id=$(sbatch \
    --dependency=afterany:${ovl_array_id} \
    "$DIR/experiment_merge.sh" "$PROJECT_ROOT" "$OVL_PENDING" "$OVL_OUTPUT" \
    | awk '{print $4}')
echo "Overlapping merge submitted — job $ovl_merge_id (after $ovl_array_id)"

# ── 3. Disjoint: 250 COPs, array 0-249 ───────────────────────────────────────
dis_array_id=$(sbatch \
    --dependency=afterok:${vcg_id} \
    --array=0-249 \
    "$DIR/experiment_array.sh" "$DIS_PARAMS" "$DIS_PENDING" \
    | awk '{print $4}')
echo "Disjoint array submitted — job $dis_array_id (COPs 0-249, after $vcg_id)"

dis_merge_id=$(sbatch \
    --dependency=afterany:${dis_array_id} \
    "$DIR/experiment_merge.sh" "$PROJECT_ROOT" "$DIS_PENDING" "$DIS_OUTPUT" \
    | awk '{print $4}')
echo "Disjoint merge submitted — job $dis_merge_id (after $dis_array_id)"

echo ""
echo "All jobs queued. Monitor with: squeue -u \$USER"
echo "Outputs:"
echo "  $OVL_OUTPUT"
echo "  $DIS_OUTPUT"
