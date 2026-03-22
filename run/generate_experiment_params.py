"""
generate_experiment_params.py -- Enumerate HybridQAOA vs PenaltyQAOA experiment
parameter combinations and write them as JSON-per-line for SLURM array jobs.

Problem structure
-----------------
Each experiment consists of 2–3 constraints drawn from any supported family,
assigned to disjoint variable ranges so that n_x = total variables <= 10.
Structural vs penalty partitioning is decided at run time by
``partition_constraints(strategy="auto")`` in the run script — not here.

Constraint families included
-----------------------------
  - cardinality        : sum x_i op b
  - knapsack           : sum a_i x_i <= b
  - quadratic_knapsack : sum Q_ij x_i x_j <= b
  - flow               : sum_in x_i - sum_out x_j == 0
  - assignment         : sum_j x_{i*n+j} == 1  (per-row equality)
  - independent_set    : x_i * x_j == 0  (per-edge equality)

Variable assignment
-------------------
Constraints are normalised to x_0..x_{k-1} when loaded, then remapped to
sequential non-overlapping ranges when building each task:
  constraint 0  → x_0 .. x_{n0-1}
  constraint 1  → x_{n0} .. x_{n0+n1-1}
  ...
n_x = sum of all constraint variable counts.

Output format (one JSON object per line)
-----------------------------------------
{
  "constraints" : ["x_0 + x_1 + x_2 == 1", "3*x_3 + 2*x_4 <= 4"],
  "families"    : ["cardinality", "knapsack"],
  "n_x"         : 5,
  "qubo_idx"    : 2,
}

Usage
-----
    python run/generate_experiment_params.py \\
        --output run/params/experiment_params.jsonl \\
        --max-tasks 500 --seed 42
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import json
import argparse
import random
import re

from analyze_results.results_helper import read_typed_csv, remap_constraint_to_vars
from data.make_data import read_qubos_from_file


# ─────────────────────────────────────────────────────────────────────────────
# Constraint pool loading
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_constraint(c: str):
    """Remap a constraint's variables to x_0, x_1, ... and return (n_vars, normalized_str)."""
    var_ids = sorted(set(int(m) for m in re.findall(r'x_(\d+)', c)))
    if not var_ids:
        return 0, c
    remap = {old: new for new, old in enumerate(var_ids)}
    normalized = re.sub(r'x_(\d+)', lambda m: f'x_{remap[int(m.group(1))]}', c)
    return len(var_ids), normalized


def _load_all_constraints(data_dir: str) -> list:
    """Return list of (n_vars, constraint_str, family) for all constraints.

    Each constraint is normalised to x_0..x_{n-1}.  Multi-constraint CSV rows
    (assignment, independent_set) contribute one entry per individual constraint.
    Only constraints with n_vars >= 2 are included.
    """
    pool = []
    sources = [
        ('cardinality',        'cardinality_constraints.csv'),
        ('knapsack',           'knapsack_constraints.csv'),
        ('quadratic_knapsack', 'quadratic_knapsack_constraints.csv'),
        ('flow',               'flow_constraints.csv'),
        ('assignment',         'assignment_constraints.csv'),
        ('independent_set',    'independent_set_constraints.csv'),
    ]
    seen = set()
    for family, fname in sources:
        csv_path = os.path.join(data_dir, fname)
        if not os.path.exists(csv_path):
            continue
        for _n_vars_row, cs in read_typed_csv(csv_path):
            for c in cs:
                n_actual, c_norm = _normalize_constraint(c)
                if n_actual < 2:
                    continue
                key = (c_norm, family)
                if key in seen:
                    continue
                seen.add(key)
                pool.append((n_actual, c_norm, family))
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Task generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_tasks(data_dir: str = 'data/', max_tasks: int = 500,
                   seed: int = 42, n_constraints_range: tuple = (2, 3),
                   max_n_x: int = 10) -> list:
    """Sample experiment tasks from all constraint families.

    Each task has 2–3 constraints assigned to disjoint variable ranges.
    Structural vs penalty partitioning is left to the run script.

    Returns list of task dicts (see module docstring for format).
    """
    rng = random.Random(seed)

    pool  = _load_all_constraints(data_dir)
    qubos = read_qubos_from_file('qubos.csv', results_dir=data_dir)

    if not pool:
        raise RuntimeError('No constraints found — check data_dir.')

    seen_keys = set()
    tasks = []
    max_attempts = max_tasks * 50

    for _ in range(max_attempts):
        if len(tasks) >= max_tasks:
            break

        n_c = rng.randint(*n_constraints_range)
        sampled = rng.sample(pool, min(n_c, len(pool)))

        # Assign sequential, disjoint variable ranges
        offset = 0
        constraint_strs = []
        families = []
        for n_vars, c_norm, fam in sampled:
            target = list(range(offset, offset + n_vars))
            c_mapped = remap_constraint_to_vars(c_norm, target)
            constraint_strs.append(c_mapped)
            families.append(fam)
            offset += n_vars

        n_x = offset
        if n_x > max_n_x or n_x < 2:
            continue
        if n_x not in qubos or not qubos[n_x]:
            continue

        key = tuple(sorted(constraint_strs))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        qubo_idx = rng.randint(0, len(qubos[n_x]) - 1)
        tasks.append({
            'constraints': constraint_strs,
            'families':    families,
            'n_x':         n_x,
            'qubo_idx':    qubo_idx,
        })

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate HybridQAOA vs PenaltyQAOA experiment parameters.'
    )
    p.add_argument('--data-dir', default='data/')
    p.add_argument('--output', default='run/params/experiment_params.jsonl',
                   help='Output JSON-lines file.')
    p.add_argument('--max-tasks', type=int, default=500,
                   help='Maximum number of tasks.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-n-x', type=int, default=10,
                   help='Maximum total variable count per task.')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    tasks = generate_tasks(
        data_dir=args.data_dir,
        max_tasks=args.max_tasks,
        seed=args.seed,
        max_n_x=args.max_n_x,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    print(f'Generated {len(tasks)} tasks → {args.output}')

    from collections import Counter
    fam_counts = Counter(f for t in tasks for f in t['families'])
    nx_counts  = Counter(t['n_x'] for t in tasks)
    print('Family breakdown:', dict(sorted(fam_counts.items())))
    print('n_x distribution:', dict(sorted(nx_counts.items())))
