#!/bin/bash
# Submit the full constraint-gadget pipeline in dependency order:
#   vcg_training → vcg_merge → experiment_array → experiment_merge
#
# Usage:
#   bash slurm/submit_all.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"

VCG_PARAMS="$PROJECT_ROOT/run/params/vcg_params.jsonl"
EXP_PARAMS="$PROJECT_ROOT/run/params/experiment_params.jsonl"

mkdir -p "$DIR/job_files"
mkdir -p "$PROJECT_ROOT/gadgets/pending"
mkdir -p "$PROJECT_ROOT/results/pending"

# ── Step 1: Generate VCG task list ───────────────────────────────────────────
echo "Generating VCG params..."
python3.11 "$PROJECT_ROOT/run/create_vcg_database.py" \
    --generate-params \
    --params-out "$VCG_PARAMS"
N_VCG=$(wc -l < "$VCG_PARAMS")
echo "VCG tasks: $N_VCG"

# ── Step 2: Submit VCG training array ────────────────────────────────────────
vcg_jobid=$(sbatch --array=0-$((N_VCG - 1)) \
    "$DIR/vcg_array.sh" "$VCG_PARAMS" \
    | awk '{print $4}')
echo "VCG training submitted: job $vcg_jobid ($N_VCG tasks)"

# ── Step 3: Merge VCG results (after all training tasks finish) ───────────────
# afterany: merge runs even if individual tasks failed; missing pickles are
# skipped. Check the merge output for gaps and resubmit failed task IDs.
vcg_merge_jobid=$(sbatch \
    --dependency=afterany:${vcg_jobid} \
    "$DIR/vcg_merge.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "VCG merge submitted: job $vcg_merge_jobid (after $vcg_jobid)"

# ── Step 4: Generate experiment task list ─────────────────────────────────────
# Regenerates experiment_params.jsonl on the server (uses data/ paths local to
# this machine). The file may already exist from a local run but regenerating
# ensures it reflects the server's data/ directory. This does NOT depend on
# VCG results — it only reads constraint CSVs and qubos.csv.
echo "Generating experiment params..."
python3.11 "$PROJECT_ROOT/run/generate_experiment_params.py" \
    --output "$EXP_PARAMS" \
    --max-tasks 500 \
    --data-dir "$PROJECT_ROOT/data/"
N_EXP=$(wc -l < "$EXP_PARAMS")
echo "Experiment tasks: $N_EXP"

# ── Step 5: Submit experiment array (after VCG merge) ────────────────────────
exp_jobid=$(sbatch \
    --dependency=afterok:${vcg_merge_jobid} \
    --array=0-$((N_EXP - 1)) \
    "$DIR/experiment_array.sh" "$EXP_PARAMS" \
    | awk '{print $4}')
echo "Experiment array submitted: job $exp_jobid ($N_EXP tasks, after $vcg_merge_jobid)"

# ── Step 6: Merge experiment results (after all experiment tasks finish) ───────
# afterany: same rationale as VCG merge above.
exp_merge_jobid=$(sbatch \
    --dependency=afterany:${exp_jobid} \
    "$DIR/experiment_merge.sh" "$PROJECT_ROOT" \
    | awk '{print $4}')
echo "Experiment merge submitted: job $exp_merge_jobid (after $exp_jobid)"

echo ""
echo "Full pipeline queued."
echo "Final results: $PROJECT_ROOT/results/hybrid_vs_penalty.pkl"
