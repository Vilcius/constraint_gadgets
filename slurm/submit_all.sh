#!/bin/bash
# Submit the full constraint-gadget pipeline via SLURM.
# Nothing runs on the login node — param generation and all work goes through jobs.
#
# Dependency chain:
#   vcg_launcher → [vcg_array → vcg_merge] → experiment_launcher → [exp_array → exp_merge]
#
# Usage (from project root):
#   bash slurm/submit_all.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"

mkdir -p "$DIR/job_files"
mkdir -p "$PROJECT_ROOT/gadgets/pending"
mkdir -p "$PROJECT_ROOT/results/pending"

vcg_launcher_id=$(sbatch "$DIR/vcg_launcher.sh" "$PROJECT_ROOT" | awk '{print $4}')
echo "VCG launcher submitted: job $vcg_launcher_id"
echo "(vcg_launcher will submit the training array, merge, and experiment launcher automatically)"
