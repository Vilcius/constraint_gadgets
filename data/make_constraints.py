"""
make_constraints.py -- Generate typed constraint CSV files.

Generates data files for seven constraint families:
  - cardinality          : sum x_i op b
  - knapsack             : sum a_i x_i <= b  (single and pairs)
  - flow                 : conservation  sum_in x_i - sum_out x_j == 0
  - subtour              : TSP subtour elimination + assignment constraints
  - assignment           : sum_j x_{i*n+j} == 1 (row) and sum_i x_{i*n+j} == 1 (col)
  - independent_set      : x_i*x_j == 0 for each edge (quadratic equality)
  - quadratic_knapsack   : sum Q_ij x_i x_j <= b (quadratic inequality)

Output format (all files):
    n_vars; ['constraint_string_1', 'constraint_string_2', ...]

Usage:
    python data/make_constraints.py                     # regenerate all
    python data/make_constraints.py --type knapsack     # just one type
    python data/make_constraints.py --type quadratic    # both quadratic types
    python data/make_constraints.py --type assignment   # assignment problem
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools as it
import numpy as np
import argparse


def check_feasibility(constraints: list, n_vars: int) -> bool:
    """Return True if at least one binary assignment satisfies all constraints."""
    for x in it.product([0, 1], repeat=n_vars):
        var_dict = {f'x_{i}': x[i] for i in range(n_vars)}
        if all(eval(c, var_dict) for c in constraints):
            return True
    return False


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------

def make_cardinality_constraints(max_n: int = 5, save_dir: str = './data/') -> None:
    """Generate sum x_i op b constraints and save to cardinality_constraints.csv.

    For n in [2, max_n], op in ['==', '<=', '>='], b in [0, n].
    Filters out infeasible instances.
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for n in range(2, max_n + 1):
        for op in ['==', '<=', '>=']:
            for b in range(0, n + 1):
                lhs = ' + '.join([f'x_{i}' for i in range(n)])
                constraint = f'{lhs} {op} {b}'
                if check_feasibility([constraint], n):
                    rows.append(f"{n}; ['{constraint}']\n")
    path = os.path.join(save_dir, 'cardinality_constraints.csv')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} cardinality constraints to {path}")


# ---------------------------------------------------------------------------
# Knapsack
# ---------------------------------------------------------------------------

def make_knapsack_constraints(max_n: int = 5, n_instances: int = 10, save_dir: str = './data/') -> None:
    """Generate random knapsack constraints and all pairwise combinations.

    Coefficients drawn from [1, 10].  RHS set to a random integer in
    [sum(a)/3, 2*sum(a)/3].  Infeasible instances are discarded.

    Writes:
        data/knapsack_constraints.csv
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for n in range(2, max_n + 1):
        count = 0
        attempts = 0
        while count < n_instances and attempts < 10 * n_instances:
            attempts += 1
            a = np.random.randint(1, 11, size=n)
            s = int(np.sum(a))
            lo = max(1, int(s / 3))
            hi = int(2 * s / 3) + 1
            if lo >= hi:
                hi = lo + 1
            rhs = int(np.random.randint(lo, hi))
            lhs = ' + '.join([f'{a[i]}*x_{i}' for i in range(n)])
            constraint = f'{lhs} <= {rhs}'
            if check_feasibility([constraint], n):
                rows.append(f"{n}; ['{constraint}']\n")
                count += 1
        if count < n_instances:
            print(f"Warning: only generated {count}/{n_instances} feasible instances for n={n}")

    path = os.path.join(save_dir, 'knapsack_constraints.csv')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} knapsack constraints to {path}")


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

def make_flow_constraints(max_in: int = 3, max_out: int = 3, save_dir: str = './data/') -> None:
    """Generate flow conservation constraints and save to flow_constraints.csv.

    Constraint form: x_0 + ... + x_{n_in-1} - x_{n_in} - ... - x_{n_vars-1} == 0
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for n_in in range(1, max_in + 1):
        for n_out in range(1, max_out + 1):
            n_vars = n_in + n_out
            in_terms = ' + '.join([f'x_{i}' for i in range(n_in)])
            out_vars = [f'x_{n_in + j}' for j in range(n_out)]
            # Build: in_terms - x_{n_in} - x_{n_in+1} - ...
            constraint = in_terms + ' - ' + ' - '.join(out_vars) + ' == 0'
            rows.append(f"{n_vars}; ['{constraint}']\n")
    path = os.path.join(save_dir, 'flow_constraints.csv')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} flow constraints to {path}")


# ---------------------------------------------------------------------------
# Subtour elimination
# ---------------------------------------------------------------------------

def make_subtour_constraints(max_cities: int = 3, save_dir: str = './data/') -> None:
    """Generate TSP subtour elimination + assignment constraints.

    Variable encoding: x_{i*k+j} is the edge i→j.  n_vars = k^2.

    For each k in [3, max_cities]:
      - Subtour: sum_{i,j in S, i≠j} x_{i*k+j} <= |S| - 1  for all proper subsets S
      - Assignment: sum_j x_{i*k+j} == 1  for each city i
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for k in range(3, max_cities + 1):
        n_vars = k * k
        constraints = []
        cities = list(range(k))

        # Subtour elimination for subsets of size 2 to k-1
        for s in range(2, k):
            for S in it.combinations(cities, s):
                terms = [f'x_{i * k + j}' for i in S for j in S if i != j]
                if terms:
                    constraints.append(' + '.join(terms) + f' <= {s - 1}')

        # Assignment: each city has exactly one outgoing edge
        for i in range(k):
            terms = [f'x_{i * k + j}' for j in range(k)]
            constraints.append(' + '.join(terms) + ' == 1')

        constraint_list_str = "['" + "', '".join(constraints) + "']"
        rows.append(f"{n_vars}; {constraint_list_str}\n")

    path = os.path.join(save_dir, 'subtour_constraints.csv')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} subtour constraint sets to {path}")


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def make_assignment_constraints(max_n: int = 3, save_dir: str = './data/') -> None:
    """Generate n×n assignment problem constraints and save to assignment_constraints.csv.

    For each n in [2, max_n], the assignment problem has n^2 binary variables
    x_{i*n+j} (1 if item i is assigned to slot j) with constraints:
      - Row: sum_j x_{i*n+j} == 1  for each row i   (item assigned to exactly one slot)
      - Col: sum_i x_{i*n+j} == 1  for each col j   (slot holds exactly one item)

    These are always feasible (any permutation matrix is a solution).

    Writes:
        data/assignment_constraints.csv
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for n in range(2, max_n + 1):
        n_vars = n * n
        constraints = []
        # Row constraints: each item assigned to exactly one slot
        for i in range(n):
            terms = [f'x_{i * n + j}' for j in range(n)]
            constraints.append(' + '.join(terms) + ' == 1')
        # Column constraints: each slot holds exactly one item
        for j in range(n):
            terms = [f'x_{i * n + j}' for i in range(n)]
            constraints.append(' + '.join(terms) + ' == 1')
        constraint_list_str = "['" + "', '".join(constraints) + "']"
        rows.append(f"{n_vars}; {constraint_list_str}\n")
    path = os.path.join(save_dir, 'assignment_constraints.csv')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} assignment constraint sets to {path}")


# ---------------------------------------------------------------------------
# Quadratic constraints
# ---------------------------------------------------------------------------

def make_independent_set_constraints(
    max_n: int = 5, n_instances: int = 10, save_dir: str = './data/'
) -> None:
    """Generate independent-set constraints (one row per graph) and save.

    For each n in [3, max_n], sample n_instances random Erdős–Rényi graphs
    (p=0.5).  Each graph produces one row with multiple equality constraints
    x_i*x_j == 0 for every edge (i, j).  Infeasible instances (cliques with
    no independent set) are discarded.

    Writes:
        data/independent_set_constraints.csv
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for n in range(3, max_n + 1):
        count = 0
        attempts = 0
        while count < n_instances and attempts < 10 * n_instances:
            attempts += 1
            # Sample random graph edges (upper-triangular, p=0.5)
            edges = [
                (i, j)
                for i in range(n)
                for j in range(i + 1, n)
                if np.random.rand() < 0.5
            ]
            if not edges:
                # No edges → trivially feasible, but uninteresting; skip
                continue
            constraints = [f'x_{i}*x_{j} == 0' for i, j in edges]
            if check_feasibility(constraints, n):
                constraint_list_str = "['" + "', '".join(constraints) + "']"
                rows.append(f"{n}; {constraint_list_str}\n")
                count += 1
        if count < n_instances:
            print(
                f"Warning: only generated {count}/{n_instances} feasible "
                f"independent-set instances for n={n}"
            )
    path = os.path.join(save_dir, 'independent_set_constraints.csv')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} independent-set constraint rows to {path}")


def make_quadratic_knapsack_constraints(
    max_n: int = 5, n_instances: int = 10, save_dir: str = './data/'
) -> None:
    """Generate random quadratic knapsack constraints and save.

    For each n in [3, max_n], sample n_instances upper-triangular Q matrices
    with Q_ij in [1, 5].  RHS is set to a random integer in
    [sum(Q)/3, 2*sum(Q)/3].  Infeasible instances are discarded.

    Writes:
        data/quadratic_knapsack_constraints.csv
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for n in range(3, max_n + 1):
        count = 0
        attempts = 0
        while count < n_instances and attempts < 10 * n_instances:
            attempts += 1
            # Upper-triangular Q (includes diagonal)
            Q = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i, n):
                    Q[i, j] = np.random.randint(1, 6)
            total = int(np.sum(Q))
            lo = max(1, total // 3)
            hi = (2 * total) // 3 + 1
            if lo >= hi:
                hi = lo + 1
            rhs = int(np.random.randint(lo, hi))
            # Build constraint string: a*x_i*x_j + ... <= rhs
            terms = []
            for i in range(n):
                for j in range(i, n):
                    if Q[i, j] != 0:
                        if i == j:
                            terms.append(f'{Q[i,j]}*x_{i}*x_{i}')
                        else:
                            terms.append(f'{Q[i,j]}*x_{i}*x_{j}')
            constraint = ' + '.join(terms) + f' <= {rhs}'
            if check_feasibility([constraint], n):
                rows.append(f"{n}; ['{constraint}']\n")
                count += 1
        if count < n_instances:
            print(
                f"Warning: only generated {count}/{n_instances} feasible "
                f"quadratic knapsack instances for n={n}"
            )
    path = os.path.join(save_dir, 'quadratic_knapsack_constraints.csv')
    with open(path, 'w') as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} quadratic knapsack constraints to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate constraint CSV files.')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'cardinality', 'knapsack', 'flow', 'subtour',
                                 'assignment', 'independent_set', 'quadratic_knapsack', 'quadratic'],
                        help='Type of constraints to generate (default: all)')
    parser.add_argument('--save_dir', type=str, default='./data/',
                        help='Directory to save constraint files')
    parser.add_argument('--max_n', type=int, default=5,
                        help='Maximum number of variables (cardinality/knapsack)')
    parser.add_argument('--n_instances', type=int, default=10,
                        help='Number of random instances per n (knapsack)')
    parser.add_argument('--max_in', type=int, default=3,
                        help='Maximum number of in-flow variables (flow)')
    parser.add_argument('--max_out', type=int, default=3,
                        help='Maximum number of out-flow variables (flow)')
    parser.add_argument('--max_cities', type=int, default=4,
                        help='Maximum number of cities (subtour)')
    args = parser.parse_args()

    if args.type in ('all', 'cardinality'):
        make_cardinality_constraints(max_n=args.max_n, save_dir=args.save_dir)
    if args.type in ('all', 'knapsack'):
        make_knapsack_constraints(max_n=args.max_n, n_instances=args.n_instances, save_dir=args.save_dir)
    if args.type in ('all', 'flow'):
        make_flow_constraints(max_in=args.max_in, max_out=args.max_out, save_dir=args.save_dir)
    if args.type in ('all', 'subtour'):
        make_subtour_constraints(max_cities=args.max_cities, save_dir=args.save_dir)
    if args.type in ('all', 'assignment'):
        make_assignment_constraints(max_n=args.max_n, save_dir=args.save_dir)
    if args.type in ('all', 'quadratic', 'independent_set'):
        make_independent_set_constraints(max_n=args.max_n, n_instances=args.n_instances, save_dir=args.save_dir)
    if args.type in ('all', 'quadratic', 'quadratic_knapsack'):
        make_quadratic_knapsack_constraints(max_n=args.max_n, n_instances=args.n_instances, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
