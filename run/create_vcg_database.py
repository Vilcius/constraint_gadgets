"""
create_vcg_database.py -- Populate the VCG gadget database for all
non-Dicke/non-Flow constraints in the project (support >= 3 variables).

Covered families
----------------
  - knapsack          (data/knapsack_constraints.csv,          n >= 3)
  - quadratic_knapsack (data/quadratic_knapsack_constraints.csv, n >= 3)

These are the only families that produce VCG-needing constraints with
support >= 3 after filtering.  Dicke/Flow families (cardinality, flow,
assignment) are handled by exact structural circuits.  The subtour and
independent-set families have their VCG-needing constraints with support=2
and are excluded.

Modes
-----
Sequential (default) -- process all constraints one by one:
    python run/create_vcg_database.py

SLURM parallel:
    # Step 1 – generate a task list
    python run/create_vcg_database.py --generate-params \\
        --params-out run/params/vcg_params.jsonl

    # Step 2 – submit an array job; each task processes one constraint
    #   sbatch --array=0-<N-1> slurm/vcg_array.sh
    #   (inside the job: python run/create_vcg_database.py --task-id $SLURM_ARRAY_TASK_ID)
    python run/create_vcg_database.py --task-id 7 \\
        --params-out run/params/vcg_params.jsonl \\
        --db gadgets/gadget_db.pkl \\
        --pending-dir gadgets/pending/

    # Step 3 – merge all pending per-task results into the main DB
    python run/create_vcg_database.py --merge \\
        --pending-dir gadgets/pending/ \\
        --db gadgets/gadget_db.pkl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import json
import argparse
import glob

import pandas as pd

from analyze_results.results_helper import read_typed_csv, GadgetDatabase
from run.add_to_vcg_database import train_and_add


# ─────────────────────────────────────────────────────────────────────────────
# Constraint discovery
# ─────────────────────────────────────────────────────────────────────────────

def _load_vcg_tasks(data_dir: str = 'data/') -> list:
    """Return a list of task dicts, one per VCG-needing constraint.

    Each dict has:
        constraints : list[str]   -- zero-indexed constraint strings
        n_vars      : int         -- number of decision variables
        family      : str         -- e.g. 'knapsack', 'quadratic_knapsack'
    """
    tasks = []
    sources = [
        ('knapsack',           'knapsack_constraints.csv'),
        ('quadratic_knapsack', 'quadratic_knapsack_constraints.csv'),
    ]
    for family, fname in sources:
        csv_path = os.path.join(data_dir, fname)
        if not os.path.exists(csv_path):
            print(f'  [warn] Not found: {csv_path}')
            continue
        for n_vars, constraints in read_typed_csv(csv_path):
            if n_vars < 3:
                continue  # support < 3: skip
            tasks.append({
                'constraints': constraints,
                'n_vars': n_vars,
                'family': family,
            })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def _merge(pending_dir: str, db_path: str) -> None:
    """Combine all per-task pending pickles into the main GadgetDatabase."""
    pattern = os.path.join(pending_dir, 'task_*.pkl')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No pending files found in {pending_dir}')
        return

    db = GadgetDatabase()
    db.load(db_path)
    added = 0
    for fpath in files:
        try:
            df = pd.read_pickle(fpath)
            for _, row in df.iterrows():
                before = len(db)
                db.add(row.to_dict())
                if len(db) > before:
                    added += 1
        except Exception as e:
            print(f'  [warn] Could not read {fpath}: {e}')
    db.save(db_path)
    print(f'Merged {len(files)} files. Added {added} new gadgets. DB now has {len(db)} entries.')
    print(f'Saved: {db_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Populate VCG gadget database for all knapsack / quadratic-knapsack constraints.'
    )
    p.add_argument('--data-dir', default='data/')
    p.add_argument('--db', default='gadgets/gadget_db.pkl',
                   help='Path to the GadgetDatabase pickle.')
    p.add_argument('--pending-dir', default='gadgets/pending/',
                   help='Directory for per-task result pickles (SLURM mode).')
    p.add_argument('--params-out', default='run/params/vcg_params.jsonl',
                   help='Path to write/read the task list (SLURM mode).')

    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--generate-params', action='store_true',
                      help='Write task list to --params-out and exit.')
    mode.add_argument('--task-id', type=int, default=None,
                      help='Process the N-th task from --params-out.')
    mode.add_argument('--merge', action='store_true',
                      help='Merge pending per-task pickles into --db.')

    # Training hyper-parameters (forwarded to train_and_add)
    p.add_argument('--ar-threshold', type=float, default=0.999)
    p.add_argument('--max-layers', type=int, default=8)
    p.add_argument('--qaoa-restarts', type=int, default=5)
    p.add_argument('--qaoa-steps', type=int, default=150)
    p.add_argument('--ma-restarts', type=int, default=20)
    p.add_argument('--ma-steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--shots', type=int, default=10_000)

    return p.parse_args()


def _train_kwargs(args) -> dict:
    return dict(
        ar_threshold=args.ar_threshold,
        max_layers=args.max_layers,
        qaoa_restarts=args.qaoa_restarts,
        qaoa_steps=args.qaoa_steps,
        ma_restarts=args.ma_restarts,
        ma_steps=args.ma_steps,
        lr=args.lr,
        shots=args.shots,
    )


if __name__ == '__main__':
    args = _parse_args()

    # ── Generate params mode ─────────────────────────────────────────────────
    if args.generate_params:
        tasks = _load_vcg_tasks(args.data_dir)
        os.makedirs(os.path.dirname(args.params_out) if os.path.dirname(args.params_out) else '.', exist_ok=True)
        with open(args.params_out, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
        print(f'Wrote {len(tasks)} tasks to {args.params_out}')
        sys.exit(0)

    # ── Merge mode ───────────────────────────────────────────────────────────
    if args.merge:
        _merge(args.pending_dir, args.db)
        sys.exit(0)

    # ── Single task (SLURM) ──────────────────────────────────────────────────
    if args.task_id is not None:
        if not os.path.exists(args.params_out):
            print(f'Params file not found: {args.params_out}')
            print('Run with --generate-params first.')
            sys.exit(1)
        with open(args.params_out) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        if args.task_id >= len(tasks):
            print(f'task-id {args.task_id} out of range (have {len(tasks)} tasks).')
            sys.exit(1)
        task = tasks[args.task_id]
        os.makedirs(args.pending_dir, exist_ok=True)
        result_out = os.path.join(args.pending_dir, f'task_{args.task_id}.pkl')
        print(f'Task {args.task_id}/{len(tasks)-1}: '
              f'{task["family"]} n={task["n_vars"]}  '
              f'{task["constraints"][0][:60]}')
        train_and_add(
            constraints=task['constraints'],
            db_path=args.db,
            result_out=result_out,
            **_train_kwargs(args),
        )
        sys.exit(0)

    # ── Sequential mode (default) ────────────────────────────────────────────
    tasks = _load_vcg_tasks(args.data_dir)
    print(f'Found {len(tasks)} VCG tasks.')
    os.makedirs(os.path.dirname(args.db) if os.path.dirname(args.db) else '.', exist_ok=True)

    for i, task in enumerate(tasks):
        print(f'\n[{i+1}/{len(tasks)}] {task["family"]} n={task["n_vars"]}')
        train_and_add(
            constraints=task['constraints'],
            db_path=args.db,
            **_train_kwargs(args),
        )

    print(f'\nDone. DB saved to {args.db}')
