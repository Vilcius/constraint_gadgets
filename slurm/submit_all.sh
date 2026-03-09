#!/bin/bash
# Submit the full constraint-gadget pipeline via SLURM.
#
# Steps 1 and 4 use --wait so the param file is ready before the array is sized.
# Everything else is submitted with dependencies.
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
sbatch --wait "$DIR/generate_vcg_params.sh" "$PROJECT_ROOT"
N_VCG=$(wc -l < "$VCG_PARAMS")
echo "Step 1 done: $N_VCG VCG tasks"

# ── Step 2: Submit VCG training array ────────────────────────────────────────
vcg_array_id=$(sbatch --array=0-$((N_VCG - 1)) \
    "$DIR/vcg_array.sh" "$VCG_PARAMS" \
    | awk '{print $4}')
echo "Step 2: VCG training array submitted — job $vcg_array_id"

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
# Array size is fixed to the max-tasks cap (500); tasks beyond the actual count
# exit cleanly (handled in run_hybrid_vs_penalty.py).
N_EXP=500
exp_array_id=$(sbatch \
    --dependency=afterok:${exp_gen_id} \
    --array=0-$((N_EXP - 1)) \
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
echo "Pipeline queued. Final results: $PROJECT_ROOT/results/hybrid_vs_penalty.pkl"
