#!/bin/bash
# Submit the full constraint-gadget pipeline via SLURM.
# Returns immediately — all 6 steps are chained via dependencies.
#
# Usage (from project root):
#   bash slurm/submit_all.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"

VCG_PARAMS="$PROJECT_ROOT/run/params/vcg_params.jsonl"
EXP_PARAMS="$PROJECT_ROOT/run/params/experiment_params.jsonl"

mkdir -p "$DIR/job_files"
mkdir -p "$PROJECT_ROOT/gadgets/pending"
mkdir -p "$PROJECT_ROOT/results/pending"

# ── Step 1: Generate VCG task list ───────────────────────────────────────────
vcg_gen_id=$(sbatch \
    "$DIR/generate_vcg_params.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "Step 1: VCG param generation submitted — job $vcg_gen_id"

# ── Step 2: Submit VCG training array ────────────────────────────────────────
# Fixed upper bound of 100 (actual task count is ~60); tasks beyond the real
# count exit cleanly.
vcg_array_id=$(sbatch \
    --dependency=afterok:${vcg_gen_id} \
    --array=0-99 \
    "$DIR/vcg_array.sh" "$VCG_PARAMS" \
    | awk '{print $4}')
echo "Step 2: VCG training array submitted — job $vcg_array_id (after $vcg_gen_id)"

# ── Step 3: Merge VCG results ────────────────────────────────────────────────
vcg_merge_id=$(sbatch \
    --dependency=afterany:${vcg_array_id} \
    "$DIR/vcg_merge.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "Step 3: VCG merge submitted — job $vcg_merge_id (after $vcg_array_id)"

# ── Step 4: Generate experiment task list ────────────────────────────────────
exp_gen_id=$(sbatch \
    --dependency=afterok:${vcg_merge_id} \
    "$DIR/generate_experiment_params.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "Step 4: Experiment param generation submitted — job $exp_gen_id (after $vcg_merge_id)"

# ── Step 5: Submit experiment array ──────────────────────────────────────────
# Fixed upper bound of 500 (matches --max-tasks cap); tasks beyond the real
# count exit cleanly.
exp_array_id=$(sbatch \
    --dependency=afterok:${exp_gen_id} \
    --array=0-499 \
    "$DIR/experiment_array.sh" "$EXP_PARAMS" \
    | awk '{print $4}')
echo "Step 5: Experiment array submitted — job $exp_array_id (after $exp_gen_id)"

# ── Step 6: Merge experiment results ─────────────────────────────────────────
exp_merge_id=$(sbatch \
    --dependency=afterany:${exp_array_id} \
    "$DIR/experiment_merge.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "Step 6: Experiment merge submitted — job $exp_merge_id (after $exp_array_id)"

echo ""
echo "All jobs queued. Check status with: squeue -u \$USER"
echo "Final results: $PROJECT_ROOT/results/hybrid_vs_penalty.pkl"
