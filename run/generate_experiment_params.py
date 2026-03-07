"""
generate_experiment_params.py -- Enumerate HybridQAOA vs PenaltyQAOA experiment
parameter combinations and write them as JSON-per-line for SLURM array jobs.

Problem structure
-----------------
Each experiment consists of:
  - 1–2 structural constraints (disjoint variable sets, support >= 3)
      - Dicke (cardinality equality, e.g. x_0+x_1+x_2==1)
      - VCG   (knapsack or quadratic-knapsack inequality)
  - 1–2 penalty constraints (support >= 3, overlapping at least 2 structural
      variables + 1 new "free" variable per constraint)
  - One QUBO of the matching total variable count.

Variable assignment
-------------------
Structural constraints are placed sequentially:
  constraint 0  → x_0 .. x_{n0-1}
  constraint 1  → x_{n0} .. x_{n0+n1-1}

Each penalty constraint uses 2 randomly-chosen structural variables and
1 fresh variable.  Penalty constraints are always unit-coefficient
cardinality inequalities (x_a + x_b + x_c <= 1) remapped to these positions.

QUBO size = (total structural vars) + (number of penalty constraints).

Output format (one JSON object per line)
-----------------------------------------
{
  "structural_constraints": ["6*x_3 + 2*x_4 + 2*x_5 <= 3"],
  "penalty_constraints":    ["x_1 + x_3 + x_6 <= 1"],
  "structural_families":    ["knapsack"],
  "penalty_family":         "cardinality",
  "structural_indices":     [0],        // indices into all_constraints
  "penalty_indices":        [1],        // indices into all_constraints
  "n_x":                    7,
  "qubo_idx":               0,          // index into qubos[n_x]
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
import itertools
import random
import re

from analyze_results.results_helper import read_typed_csv, remap_constraint_to_vars
from data.make_data import read_qubos_from_file


# ─────────────────────────────────────────────────────────────────────────────
# Constraint pool loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_structural_pool(data_dir: str) -> list:
    """Return list of (n_vars, constraint_str, family) for all structural
    constraints with support >= 3.

    Includes:
      - Dicke  : cardinality equalities (x_0+...==1)
      - VCG    : knapsack, quadratic_knapsack inequalities
    """
    pool = []
    sources = [
        ('cardinality',        'cardinality_constraints.csv'),
        ('knapsack',           'knapsack_constraints.csv'),
        ('quadratic_knapsack', 'quadratic_knapsack_constraints.csv'),
    ]
    for family, fname in sources:
        csv_path = os.path.join(data_dir, fname)
        if not os.path.exists(csv_path):
            continue
        for n_vars, cs in read_typed_csv(csv_path):
            if n_vars < 3:
                continue
            for c in cs:
                pool.append((n_vars, c, family))
    return pool


def _load_penalty_pool(data_dir: str) -> list:
    """Return list of (n_vars, constraint_str, family) for cardinality
    INEQUALITY constraints with support >= 3.  These are used as penalty
    constraints since they have unit coefficients and are easy to remap.
    """
    pool = []
    csv_path = os.path.join(data_dir, 'cardinality_constraints.csv')
    if not os.path.exists(csv_path):
        return pool
    for n_vars, cs in read_typed_csv(csv_path):
        if n_vars < 3:
            continue
        for c in cs:
            if '<=' in c or '>=' in c:
                pool.append((n_vars, c, 'cardinality'))
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Constraint remapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _penalty_constraint_with_overlap(penalty_str: str, n_vars: int,
                                      structural_vars: list,
                                      free_var: int,
                                      rng: random.Random) -> str:
    """Remap a zero-indexed penalty constraint (n_vars vars) so that
    (n_vars-1) of its variables land on existing structural positions and
    1 lands on free_var.

    For n_vars=3 (the typical cardinality inequality), picks 2 structural
    vars and maps the third to free_var.
    """
    n_overlap = n_vars - 1  # structural vars to use
    chosen = rng.sample(structural_vars, min(n_overlap, len(structural_vars)))
    # Pad to n_vars-1 by repeating last if structural_vars is short
    while len(chosen) < n_overlap:
        chosen.append(chosen[-1])
    target = chosen + [free_var]
    return remap_constraint_to_vars(penalty_str, target)


# ─────────────────────────────────────────────────────────────────────────────
# Task generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_tasks(data_dir: str = 'data/', max_tasks: int = 500,
                   seed: int = 42) -> list:
    """Enumerate experiment tasks.

    Returns list of task dicts (see module docstring for format).
    """
    rng = random.Random(seed)

    structural_pool = _load_structural_pool(data_dir)
    penalty_pool    = _load_penalty_pool(data_dir)
    qubos           = read_qubos_from_file('qubos.csv', results_dir=data_dir)

    if not penalty_pool:
        raise RuntimeError('No penalty constraints found in cardinality_constraints.csv')

    tasks = []

    # ── 1-structural + 1-penalty ─────────────────────────────────────────────
    for n_s, c_s, fam_s in structural_pool:
        # Assign structural vars 0..n_s-1
        c_s_mapped = remap_constraint_to_vars(c_s, list(range(n_s)))
        struct_vars = list(range(n_s))
        free_var = n_s  # one new variable for the penalty constraint
        n_x = n_s + 1

        if n_x not in qubos or not qubos[n_x]:
            continue

        # Pick a random penalty constraint (n=3 cardinality inequality)
        pen_pool_3 = [(n, c, f) for n, c, f in penalty_pool if n == 3]
        if not pen_pool_3:
            continue
        n_p, c_p, fam_p = rng.choice(pen_pool_3)
        c_p_mapped = _penalty_constraint_with_overlap(
            c_p, n_p, struct_vars, free_var, rng
        )

        all_constraints = [c_s_mapped, c_p_mapped]
        qubo_idx = rng.randint(0, len(qubos[n_x]) - 1)
        tasks.append({
            'structural_constraints': [c_s_mapped],
            'penalty_constraints':    [c_p_mapped],
            'structural_families':    [fam_s],
            'penalty_family':         fam_p,
            'structural_indices':     [0],
            'penalty_indices':        [1],
            'n_x':                    n_x,
            'qubo_idx':               qubo_idx,
        })

    # ── 1-structural + 2-penalty ─────────────────────────────────────────────
    for n_s, c_s, fam_s in structural_pool:
        c_s_mapped = remap_constraint_to_vars(c_s, list(range(n_s)))
        struct_vars = list(range(n_s))
        n_x = n_s + 2  # two new variables

        if n_x not in qubos or not qubos[n_x] or n_x > 10:
            continue

        pen_pool_3 = [(n, c, f) for n, c, f in penalty_pool if n == 3]
        if len(pen_pool_3) < 2:
            continue
        chosen_pen = rng.sample(pen_pool_3, 2)

        pen_mapped = []
        for pi, (n_p, c_p, fam_p) in enumerate(chosen_pen):
            free_var = n_s + pi
            pen_mapped.append(
                _penalty_constraint_with_overlap(c_p, n_p, struct_vars, free_var, rng)
            )

        all_constraints = [c_s_mapped] + pen_mapped
        qubo_idx = rng.randint(0, len(qubos[n_x]) - 1)
        tasks.append({
            'structural_constraints': [c_s_mapped],
            'penalty_constraints':    pen_mapped,
            'structural_families':    [fam_s],
            'penalty_family':         chosen_pen[0][2],
            'structural_indices':     [0],
            'penalty_indices':        [1, 2],
            'n_x':                    n_x,
            'qubo_idx':               qubo_idx,
        })

    # ── 2-structural + 1-penalty ─────────────────────────────────────────────
    # Enumerate all pairs of structural constraints from DIFFERENT families
    # (or same family is fine too), where total vars <= 8 (leaves room for QUBO).
    pool_pairs = list(itertools.combinations(range(len(structural_pool)), 2))
    rng.shuffle(pool_pairs)

    for i, j in pool_pairs:
        n_s1, c_s1, fam_s1 = structural_pool[i]
        n_s2, c_s2, fam_s2 = structural_pool[j]
        total_s = n_s1 + n_s2
        n_x = total_s + 1  # one new variable for penalty

        if n_x > 10 or n_x not in qubos or not qubos[n_x]:
            continue

        # Assign vars: constraint1 → 0..n_s1-1, constraint2 → n_s1..total_s-1
        c_s1_mapped = remap_constraint_to_vars(c_s1, list(range(n_s1)))
        c_s2_mapped = remap_constraint_to_vars(c_s2, list(range(n_s1, total_s)))
        struct_vars = list(range(total_s))  # all structural vars
        free_var = total_s

        pen_pool_3 = [(n, c, f) for n, c, f in penalty_pool if n == 3]
        if not pen_pool_3:
            continue
        n_p, c_p, fam_p = rng.choice(pen_pool_3)
        c_p_mapped = _penalty_constraint_with_overlap(
            c_p, n_p, struct_vars, free_var, rng
        )

        qubo_idx = rng.randint(0, len(qubos[n_x]) - 1)
        tasks.append({
            'structural_constraints': [c_s1_mapped, c_s2_mapped],
            'penalty_constraints':    [c_p_mapped],
            'structural_families':    [fam_s1, fam_s2],
            'penalty_family':         fam_p,
            'structural_indices':     [0, 1],
            'penalty_indices':        [2],
            'n_x':                    n_x,
            'qubo_idx':               qubo_idx,
        })

    # ── Random sample if over budget ─────────────────────────────────────────
    if len(tasks) > max_tasks:
        tasks = rng.sample(tasks, max_tasks)

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
                   help='Maximum number of tasks (random sample if exceeded).')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    tasks = generate_tasks(
        data_dir=args.data_dir,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    print(f'Generated {len(tasks)} tasks → {args.output}')

    # Summary breakdown
    n1 = sum(1 for t in tasks if len(t['structural_constraints']) == 1
             and len(t['penalty_constraints']) == 1)
    n2 = sum(1 for t in tasks if len(t['structural_constraints']) == 1
             and len(t['penalty_constraints']) == 2)
    n3 = sum(1 for t in tasks if len(t['structural_constraints']) == 2)
    print(f'  1-struct + 1-penalty : {n1}')
    print(f'  1-struct + 2-penalty : {n2}')
    print(f'  2-struct + 1-penalty : {n3}')
