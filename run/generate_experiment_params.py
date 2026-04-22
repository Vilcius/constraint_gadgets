"""
generate_experiment_params.py -- Enumerate HybridQAOA vs PenaltyQAOA experiment
parameter combinations and write them as JSON-per-line for SLURM array jobs.

Problem structure
-----------------
Each constrained optimization problem (COP) consists of 2–3 constraints drawn
from any supported family, placed on a shared variable space of size n_x.
Variable assignments overlap across constraints so that
partition_constraints(strategy='auto') must make genuine structural-vs-penalty
decisions at run time.

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
COPs where all pairs happen to be disjoint despite the shared pool are rejected.

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
        --output run/params/experiment_params_overlapping.jsonl \\
        --max-cops 250 --seed 42
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
# COP generation
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


def generate_cops(data_dir: str = 'data/', max_cops: int = 250,
                  seed: int = 42, n_constraints_range: tuple = (2, 3),
                  min_n_x: int = 3, max_n_x: int = 8,
                  disjoint: bool = False) -> list:
    """Sample constrained optimization problems (COPs) from all constraint families.

    Each COP has 2–3 constraints (controlled by n_constraints_range) drawn
    from the constraint pool and mapped onto a shared variable space of
    size n_x in [min_n_x, max_n_x].  Only COPs with at least one feasible
    bitstring are included.

    Parameters
    ----------
    n_constraints_range : tuple
        (min, max) number of constraints per COP, inclusive.
    min_n_x, max_n_x : int
        Allowed range for the total variable count.  Defaults 3–8 keep
        circuit sizes manageable on the cluster.
    disjoint : bool
        If False (default), n_x is chosen strictly less than the sum of
        individual variable counts so at least two constraints must share
        at least one variable (overlapping mode).
        If True, each constraint is assigned its own non-overlapping slice
        of variables — n_x equals the sum of individual variable counts
        (disjoint mode).

    Returns list of COP dicts (see module docstring for format).
    """
    rng = random.Random(seed)

    pool  = _load_all_constraints(data_dir)
    qubos = read_qubos_from_file('qubos.csv', results_dir=data_dir)

    if not pool:
        raise RuntimeError('No constraints found — check data_dir.')

    seen_keys = set()
    cops = []
    max_attempts = max_cops * 200

    for _ in range(max_attempts):
        if len(cops) >= max_cops:
            break

        n_c = rng.randint(*n_constraints_range)
        sampled = rng.sample(pool, min(n_c, len(pool)))

        total_vars = sum(k for k, _, _ in sampled)
        max_k = max(k for k, _, _ in sampled)

        if disjoint:
            # n_x = total_vars so each constraint gets its own variables.
            n_x = total_vars
            if n_x < min_n_x or n_x > max_n_x:
                continue
            if n_x not in qubos or not qubos[n_x]:
                continue

            # Assign variables as non-overlapping consecutive slices.
            constraint_strs = []
            families = []
            offset = 0
            for k, c_norm, fam in sampled:
                chosen = list(range(offset, offset + k))
                constraint_strs.append(remap_constraint_to_vars(c_norm, chosen))
                families.append(fam)
                offset += k

        else:
            # n_x strictly less than total_vars to force at least one shared variable.
            n_x_min = max(max_k, min_n_x)
            n_x_max = min(total_vars - 1, max_n_x)
            if n_x_min > n_x_max:
                continue

            valid_nx = [n for n in range(n_x_min, n_x_max + 1) if n in qubos and qubos[n]]
            if not valid_nx:
                continue
            n_x = rng.choice(valid_nx)

            constraint_strs = []
            families = []
            for k, c_norm, fam in sampled:
                chosen = rng.sample(range(n_x), k)
                constraint_strs.append(remap_constraint_to_vars(c_norm, chosen))
                families.append(fam)

            # Reject if no pair of constraints actually shares a variable.
            var_sets = [_vars_used(c) for c in constraint_strs]
            any_overlap = any(
                not var_sets[i].isdisjoint(var_sets[j])
                for i in range(len(var_sets))
                for j in range(i + 1, len(var_sets))
            )
            if not any_overlap:
                continue

        key = tuple(sorted(constraint_strs))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if not _has_feasible_solution(constraint_strs, n_x):
            continue

        qubo_idx = rng.randint(0, len(qubos[n_x]) - 1)
        cops.append({
            'constraints': constraint_strs,
            'families':    families,
            'n_x':         n_x,
            'qubo_idx':    qubo_idx,
        })

    return cops


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate HybridQAOA vs PenaltyQAOA experiment parameters.'
    )
    p.add_argument('--data-dir', default='data/')
    p.add_argument('--output', default='run/params/experiment_params_overlapping.jsonl',
                   help='Output JSON-lines file.')
    p.add_argument('--max-cops', type=int, default=250,
                   help='Maximum number of COPs to generate.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-n-x', type=int, default=3,
                   help='Minimum variable count per COP.')
    p.add_argument('--max-n-x', type=int, default=8,
                   help='Maximum variable count per COP.')
    p.add_argument('--disjoint', action='store_true',
                   help='Generate COPs with fully disjoint constraint variable supports.')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    # Auto-set output path based on --disjoint if user didn't override
    if args.disjoint and args.output == 'run/params/experiment_params_overlapping.jsonl':
        args.output = 'run/params/experiment_params_disjoint.jsonl'
    cops = generate_cops(
        data_dir=args.data_dir,
        max_cops=args.max_cops,
        seed=args.seed,
        min_n_x=args.min_n_x,
        max_n_x=args.max_n_x,
        disjoint=args.disjoint,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        for cop in cops:
            f.write(json.dumps(cop) + '\n')

    print(f'Generated {len(cops)} COPs → {args.output}')

    from collections import Counter
    fam_counts = Counter(f for cop in cops for f in cop['families'])
    nx_counts  = Counter(cop['n_x'] for cop in cops)
    print('Family breakdown:', dict(sorted(fam_counts.items())))
    print('n_x distribution:', dict(sorted(nx_counts.items())))

    # Show partitioning preview on a sample
    from core import constraint_handler as ch
    n_show = min(5, len(cops))
    print(f'\nPartitioning preview (first {n_show} COPs):')
    for cop in cops[:n_show]:
        parsed = ch.parse_constraints(cop['constraints'])
        si, pi = ch.partition_constraints(parsed, strategy='auto')
        print(f"  n_x={cop['n_x']}  families={cop['families']}")
        print(f"    structural: {[cop['constraints'][i] for i in si]}")
        print(f"    penalty:    {[cop['constraints'][i] for i in pi]}")
