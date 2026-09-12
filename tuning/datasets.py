"""Small seeded extensions around the repository's existing generators."""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import io
import itertools
import json
from math import comb
from pathlib import Path
import tempfile
import time

import numpy as np

from core import constraint_handler as ch
from data.make_constraints import make_knapsack_constraints, make_quadratic_knapsack_constraints
from data.make_data import generate_random_qubo_string
from run.generate_experiment_params import generate_cops
from analyze_results.results_helper import read_typed_csv
from storage import ROOT, digest, write_json

FAMILIES = ['positive', 'mixed', 'equality', 'quadratic']


@contextmanager
def seeded_numpy(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def bitstrings(n):
    return [format(i, f'0{n}b') for i in range(2**n)]


def feasible_mask(constraints, n):
    parsed = ch.parse_constraints(constraints)
    return np.asarray([ch.check_feasibility(bs, parsed, n) for bs in bitstrings(n)])


def exact_case(constraint, mask, n):
    parsed = ch.parse_constraint(constraint)
    if any(check(parsed) for check in [ch.is_dicke_compatible,
           ch.is_cardinality_leq_compatible, ch.is_cardinality_geq_single_compatible,
           ch.is_independent_set_pair_compatible, ch.is_flow_compatible]):
        return True
    if int(mask.sum()) <= 1 or mask.all():
        return True
    by_weight = Counter(bs.count('1') for bs, yes in zip(bitstrings(n), mask) if yes)
    return all(count == comb(n, weight) for weight, count in by_weight.items())


def linear_candidate(rng, n, family):
    coefficients = rng.integers(1, 6, size=n)
    if family == 'mixed':
        signs = rng.choice([-1, 1], size=n)
        if np.all(signs == signs[0]):
            return None
        coefficients *= signs
        low, high = int(coefficients[coefficients < 0].sum()), int(coefficients[coefficients > 0].sum())
        rhs = int(rng.integers(low + (high-low)//3, low + 2*(high-low)//3 + 1))
        operator = '<='
    else:
        total = int(coefficients.sum())
        values = sorted({sum(c*b for c, b in zip(coefficients, bits))
                         for bits in itertools.product([0, 1], repeat=n)})
        allowed = [int(v) for v in values if max(1, total//3) <= v <= 2*total//3]
        if not allowed:
            return None
        rhs, operator = int(rng.choice(allowed)), '=='
    lhs = ' + '.join(f'{int(c)}*x_{i}' for i, c in enumerate(coefficients))
    return lhs.replace('+ -', '- ') + f' {operator} {rhs}'


def generate_a(config):
    settings = config['study_a']
    rng = np.random.default_rng(config['dataset_seed'])
    quotas = {}
    for idx, n in enumerate(settings['supports']):
        count = settings['per_support']
        quotas[n] = {f: count//4 for f in FAMILIES}
        for extra in range(count % 4):
            quotas[n][FAMILIES[(idx*(count % 4)+extra) % 4]] += 1
    rows, seen, counts = [], set(), Counter()

    def accept(n, family, constraint):
        if not constraint or n not in quotas or counts[n, family] >= quotas[n][family]:
            return
        constraint = ch.normalize_constraint(constraint)
        mask = feasible_mask([constraint], n)
        key = (n, tuple(mask.tolist()))
        if key in seen or exact_case(constraint, mask, n):
            return
        seen.add(key)
        counts[n, family] += 1
        rows.append(dict(id=f'a_{n}_{family}_{counts[n, family]:02d}', support=n,
                         family=family, constraint=constraint,
                         feasible_mask=mask.tolist(), feasible_count=int(mask.sum()),
                         feasible_fraction=float(mask.mean())))

    target = sum(sum(q.values()) for q in quotas.values())
    with tempfile.TemporaryDirectory(prefix='pcqaoa-candidates-') as directory:
        for batch in range(100):
            with seeded_numpy(config['dataset_seed'] + batch), redirect_stdout(io.StringIO()):
                make_knapsack_constraints(max(settings['supports']), 30, directory)
                make_quadratic_knapsack_constraints(max(settings['supports']), 30, directory)
            for family, filename in [('positive', 'knapsack_constraints.csv'),
                                     ('quadratic', 'quadratic_knapsack_constraints.csv')]:
                for n, constraints in read_typed_csv(str(Path(directory) / filename)):
                    for constraint in constraints:
                        accept(int(n), family, constraint)
            for n in settings['supports']:
                for family in ['mixed', 'equality']:
                    for _ in range(100):
                        if counts[n, family] >= quotas[n][family]:
                            break
                        accept(n, family, linear_candidate(rng, n, family))
            if len(rows) == target:
                break
    if len(rows) != target:
        raise RuntimeError(f'Could not satisfy Study A quotas: {dict(counts)} / {quotas}')
    return sorted(rows, key=lambda row: row['id'])


def objective_reference(Q, constraints):
    n = len(Q)
    strings = bitstrings(n)
    bits = np.asarray([[int(c) for c in bs] for bs in strings])
    values = np.einsum('bi,ij,bj->b', bits, Q, bits)
    mask = feasible_mask(constraints, n)
    if not mask.any():
        raise ValueError('Infeasible COP')
    best = float(values[mask].min())
    return dict(feasible_mask=mask.tolist(), objective_values=values.tolist(),
                optimal_indices=np.flatnonzero(mask & (values == best)).tolist(),
                f_star=best, f_max_feasible=float(values[mask].max()),
                f_min_unconstrained=float(values.min()), feasible_fraction=float(mask.mean()))


def find_feasible_point(constraints, n, seed):
    """Seeded search without replacement; stop at first feasible bitstring.

    No objective or reference-solution information enters this search.
    """
    begin = time.perf_counter()
    parsed = ch.parse_constraints(constraints)
    for attempts, index in enumerate(np.random.default_rng(seed).permutation(2**n), 1):
        bits = format(int(index), f'0{n}b')
        if ch.check_feasibility(bits, parsed, n):
            return dict(feasible_point=bits, feasibility_search_seed=seed,
                        feasibility_search_attempts=attempts,
                        feasibility_search_seconds=time.perf_counter()-begin)
    raise ValueError('No feasible point exists')


def generate_b(config):
    settings = config['study_b']
    rows, selected_constraints = [], set()
    family_counts, partition_counts, number_counts = Counter(), Counter(), Counter()
    placeholder = {n: [{}] for n in settings['sizes']}
    for n in settings['sizes']:
        for regime, disjoint in [('disjoint', True), ('overlapping', False)]:
            candidates = generate_cops(
                data_dir=str(ROOT / 'data'), max_cops=150,
                seed=config['dataset_seed'] + 1000 + n*2 + int(disjoint),
                min_n_x=n, max_n_x=n, disjoint=disjoint, qubos=placeholder)
            for row in candidates:
                parsed = ch.parse_constraints(row['constraints'])
                structural, penalty = ch.partition_constraints(parsed)
                row['structural_indices'], row['penalty_indices'] = structural, penalty
            for index in range(settings['per_regime']):
                candidates = [c for c in candidates
                              if tuple(sorted(c['constraints'])) not in selected_constraints]
                if not candidates:
                    raise RuntimeError(f'Insufficient unique COPs at n={n}, {regime}')
                def score(c):
                    return (sum(family_counts[f] for f in set(c['families'])) /
                            len(set(c['families'])) + number_counts[len(c['constraints'])] +
                            partition_counts[len(c['structural_indices']), len(c['penalty_indices'])])
                row = min(candidates, key=score).copy()
                selected_constraints.add(tuple(sorted(row['constraints'])))
                family_counts.update(set(row['families']))
                number_counts[len(row['constraints'])] += 1
                partition_counts[len(row['structural_indices']), len(row['penalty_indices'])] += 1
                # Fresh objective for each selected COP, using the original function.
                with seeded_numpy(config['dataset_seed'] + 20000 + len(rows)):
                    Q, _ = generate_random_qubo_string(n)
                Q = np.asarray(Q, dtype=int)
                feasible_point = find_feasible_point(
                    row['constraints'], n, config['dataset_seed'] + 30000 + len(rows))
                parsed = ch.parse_constraints(row['constraints'])
                _, slack = ch.determine_slack_variables(parsed, n)
                _, pc_slack = ch.determine_slack_variables(
                    [parsed[i] for i in row['penalty_indices']], n)
                row.update(id=f'b_{n}_{regime}_{index+1:02d}', regime=regime,
                           Q=Q.tolist(), n_slack=slack, n_total=n+slack,
                           pc_n_slack=pc_slack, n_structural=len(row['structural_indices']),
                           n_penalty=len(row['penalty_indices']),
                           objective_seed=config['dataset_seed'] + 20000 + len(rows),
                           **feasible_point,
                           **objective_reference(Q, row['constraints']))
                row.pop('qubo_idx', None)
                rows.append(row)
    return rows


def ensure_manifest(output, config, study):
    path = Path(output) / f'manifest_{study}.json'
    specification = dict(seed=config['dataset_seed'], settings=config[f'study_{study}'])
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['specification'] != specification:
            raise ValueError('Dataset configuration changed; choose a new output directory')
        if manifest['hash'] != digest(manifest['instances']):
            raise ValueError('Dataset hash mismatch')
        return manifest
    instances = (generate_a if study == 'a' else generate_b)(config)
    manifest = dict(specification=specification, hash=digest(instances), instances=instances)
    write_json(path, manifest)
    return manifest
