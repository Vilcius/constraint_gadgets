"""
generate_params.py -- Generate parameter files for batch SLURM runs.

Delegates to the run/ scripts which own the actual parameter formats.

Usage
-----
    # Generate VCG database task list (one line per constraint to train):
    python run/create_vcg_database.py --generate-params \\
        --params-out run/params/vcg_params.jsonl

    # Generate experiment task list (HybridQAOA vs PenaltyQAOA):
    python run/generate_experiment_params.py \\
        --output run/params/experiment_params.jsonl \\
        --max-tasks 500

Both commands are also accessible via this script for convenience:
    python slurm/generate_params.py --vcg
    python slurm/generate_params.py --hybrid
    python slurm/generate_params.py  (both)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import argparse


def _run(cmd: list) -> None:
    print('$', ' '.join(cmd))
    subprocess.run([sys.executable] + cmd, check=True)


def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate SLURM parameter files for VCG training and experiments.'
    )
    p.add_argument('--vcg', action='store_true',
                   help='Generate VCG database task list.')
    p.add_argument('--hybrid', action='store_true',
                   help='Generate experiment task list.')
    p.add_argument('--vcg-params', default='run/params/vcg_params.jsonl')
    p.add_argument('--experiment-params', default='run/params/experiment_params.jsonl')
    p.add_argument('--max-tasks', type=int, default=500)
    p.add_argument('--data-dir', default='data/')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    do_both = not args.vcg and not args.hybrid

    if args.vcg or do_both:
        _run([
            'run/create_vcg_database.py',
            '--generate-params',
            '--params-out', args.vcg_params,
            '--data-dir', args.data_dir,
        ])

    if args.hybrid or do_both:
        _run([
            'run/generate_experiment_params.py',
            '--output', args.experiment_params,
            '--max-tasks', str(args.max_tasks),
            '--data-dir', args.data_dir,
        ])
