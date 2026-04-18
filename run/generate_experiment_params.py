"""
generate_experiment_params.py -- Enumerate HybridQAOA vs PenaltyQAOA experiment
parameter combinations and write them as JSON-per-line for SLURM array jobs.

Problem structure
-----------------
Each experiment consists of 2–3 constraints drawn from any supported family,
placed on a shared variable space of size n_x.  Variable assignments overlap
across constraints so that partition_constraints(strategy='auto') must make
genuine structural-vs-penalty decisions at run time.

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
Constraints are normalised to x_0..x_{k-1} when loaded, then each constraint's
k variables are randomly sampled (without replacement within the constraint, but
with possible overlap across constraints) from a shared pool {0, ..., n_x-1}.
n_x is chosen to be strictly less than the sum of individual variable counts,
which guarantees that at least two constraints must share at least one variable.
Tasks where all pairs happen to be disjoint despite the shared pool are rejected.

Output format (one JSON object per line)
-----------------------------------------
{
  "constraints" : ["x_0 + x_1 + x_2 == 1", "3*x_1 + 2*x_3 <= 4"],
  "families"    : ["cardinality", "knapsack"],
  "n_x"         : 4,
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

def _vars_used(constraint_str: str) -> set:
    """Return set of variable indices referenced in a constraint string."""
    return set(int(m) for m in re.findall(r'x_(\d+)', constraint_str))


def _has_feasible_solution(constraints: list, n_x: int) -> bool:
    """Return True if any bitstring of length n_x satisfies all constraints."""
    import itertools
    for bits in itertools.product([0, 1], repeat=n_x):
        var_dict = {f'x_{i}': b for i, b in enumerate(bits)}
        if all(eval(c, {"__builtins__": {}}, var_dict) for c in constraints):
            return True
    return False


def generate_tasks(data_dir: str = 'data/', max_tasks: int = 500,
                   seed: int = 42, n_constraints_range: tuple = (2, 3),
                   max_n_x: int = 10, allow_infeasible: bool = False) -> list:
    """Sample experiment tasks from all constraint families with overlapping variable supports.

    Each task has 2–3 constraints mapped onto a shared variable space of size
    n_x < sum(individual variable counts), so constraints are forced to share
    at least some variables.  Tasks where all constraints happen to be
    variable-disjoint despite the shared pool are rejected.

    The overlapping supports mean that partition_constraints(strategy='auto')
    must genuinely decide which constraints to handle structurally and which
    to penalize.

    Parameters
    ----------
    allow_infeasible : bool
        If False (default), tasks whose constraint combination has no feasible
        bitstring are silently skipped.  Pass True to include them (useful for
        studying method behaviour on infeasible instances).

    Returns list of task dicts (see module docstring for format).
    """
    rng = random.Random(seed)

    pool  = _load_all_constraints(data_dir)
    qubos = read_qubos_from_file('qubos.csv', results_dir=data_dir)

    if not pool:
        raise RuntimeError('No constraints found — check data_dir.')

    seen_keys = set()
    tasks = []
    max_attempts = max_tasks * 200

    for _ in range(max_attempts):
        if len(tasks) >= max_tasks:
            break

        n_c = rng.randint(*n_constraints_range)
        sampled = rng.sample(pool, min(n_c, len(pool)))

        # Total variables if assigned disjointly
        total_vars = sum(k for k, _, _ in sampled)
        max_k = max(k for k, _, _ in sampled)

        # Choose n_x strictly smaller than total_vars to force overlap,
        # but large enough that every individual constraint fits.
        # n_x in [max_k, total_vars - 1], capped at max_n_x.
        n_x_min = max_k
        n_x_max = min(total_vars - 1, max_n_x)
        if n_x_min > n_x_max:
            continue  # impossible to create overlap within max_n_x

        if n_x_min not in qubos and not any(n in qubos for n in range(n_x_min, n_x_max + 1)):
            continue

        # Pick a valid n_x that has QUBOs available
        valid_nx = [n for n in range(n_x_min, n_x_max + 1) if n in qubos and qubos[n]]
        if not valid_nx:
            continue
        n_x = rng.choice(valid_nx)

        # Assign each constraint's variables by sampling without replacement
        # from {0, ..., n_x-1}.  Overlaps across constraints are allowed.
        constraint_strs = []
        families = []
        for k, c_norm, fam in sampled:
            if k > n_x:
                break  # shouldn't happen given n_x >= max_k, but guard anyway
            chosen = rng.sample(range(n_x), k)
            c_mapped = remap_constraint_to_vars(c_norm, chosen)
            constraint_strs.append(c_mapped)
            families.append(fam)
        else:
            # Check that at least one pair of constraints shares a variable
            var_sets = [_vars_used(c) for c in constraint_strs]
            any_overlap = any(
                not var_sets[i].isdisjoint(var_sets[j])
                for i in range(len(var_sets))
                for j in range(i + 1, len(var_sets))
            )
            if not any_overlap and rng.random() > 0.20:
                continue  # allow ~20% fully-disjoint tasks, reject the rest

            key = tuple(sorted(constraint_strs))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if not allow_infeasible and not _has_feasible_solution(constraint_strs, n_x):
                continue

            qubo_idx = rng.randint(0, len(qubos[n_x]) - 1)
            tasks.append({
                'constraints': constraint_strs,
                'families':    families,
                'n_x':         n_x,
                'qubo_idx':    qubo_idx,
            })

        # Explicit break check in case inner loop broke early
        if len(tasks) >= max_tasks:
            break

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
    p.add_argument('--allow-infeasible', action='store_true',
                   help='Include tasks whose constraints have no feasible solution.')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    tasks = generate_tasks(
        data_dir=args.data_dir,
        max_tasks=args.max_tasks,
        seed=args.seed,
        max_n_x=args.max_n_x,
        allow_infeasible=args.allow_infeasible,
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

    # Show partitioning preview on a sample
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import constraint_handler as ch
    n_show = min(5, len(tasks))
    print(f'\nPartitioning preview (first {n_show} tasks):')
    for t in tasks[:n_show]:
        parsed = ch.parse_constraints(t['constraints'])
        si, pi = ch.partition_constraints(parsed, strategy='auto')
        print(f"  n_x={t['n_x']}  families={t['families']}")
        print(f"    structural: {[t['constraints'][i] for i in si]}")
        print(f"    penalty:    {[t['constraints'][i] for i in pi]}")
