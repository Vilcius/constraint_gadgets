"""
generate_vcg_params.py -- Write VCG training params for all knapsack and
quadratic_knapsack constraints in data/.

Cardinality, flow, assignment, and independent_set constraints are handled by
exact state preparations (Dicke/Flow) and do not need a VCG gadget.  Knapsack
and quadratic_knapsack constraints cannot be prepared exactly, so a VCG must
be trained for each unique one.

Each entry in the output file is:
    {"constraints": ["<normalized_constraint>"], "family": "<family>"}

Constraints are normalized to x_0, x_1, ... (sorted variable indices) so
the VCG database keys match the lookup performed by HybridQAOA at runtime.

Usage
-----
    python run/generate_vcg_params.py
    python run/generate_vcg_params.py --data-dir data/ --output run/params/vcg_params_experiments.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

from analyze_results.results_helper import read_typed_csv
from core.constraint_handler import normalize_constraint


_VCG_SOURCES = [
    ('knapsack',           'knapsack_constraints.csv'),
    ('quadratic_knapsack', 'quadratic_knapsack_constraints.csv'),
]


def generate_vcg_params(data_dir: str = 'data/') -> list:
    """Load all knapsack and quadratic_knapsack constraints, normalize, deduplicate.

    Returns list of dicts: {"constraints": [normalized_str], "family": family}.
    """
    tasks = []
    seen = set()
    for family, fname in _VCG_SOURCES:
        csv_path = os.path.join(data_dir, fname)
        if not os.path.exists(csv_path):
            print(f'  [warn] not found: {csv_path}')
            continue
        rows = read_typed_csv(csv_path)
        for _n_vars, constraints in rows:
            for c in constraints:
                key = normalize_constraint(c)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append({'constraints': [key], 'family': family})
        print(f'  {family}: loaded from {fname}')
    return tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data/')
    parser.add_argument('--output',   default='run/params/vcg_params_experiments.jsonl')
    args = parser.parse_args()

    tasks = generate_vcg_params(data_dir=args.data_dir)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        for t in tasks:
            f.write(json.dumps(t) + '\n')

    from collections import Counter
    fam_counts = Counter(t['family'] for t in tasks)
    print(f'\n{len(tasks)} unique VCG constraints → {args.output}')
    print('Breakdown:', dict(sorted(fam_counts.items())))
