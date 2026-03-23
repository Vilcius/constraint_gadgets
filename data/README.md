# data

Constraint datasets and QUBO generation scripts.

## Files

| File | Purpose |
|---|---|
| `make_constraints.py` | Generate all constraint CSV files |
| `make_data.py` | Generate QUBO matrices and compute optimal solutions by brute force |
| `cardinality_constraints.csv` | `sum x_i op b` constraints, n=2..5, op in {==, <=, >=} |
| `knapsack_constraints.csv` | `sum a_i x_i <= b` constraints, n=3..8 |
| `quadratic_knapsack_constraints.csv` | `sum Q_ij x_i x_j <= b` constraints, n=3..8 |
| `flow_constraints.csv` | Flow conservation `sum_in x_i - sum_out x_j == 0`, various topologies |
| `assignment_constraints.csv` | Row/column assignment `sum_j x_{i*n+j} == 1`, n=2..4 |
| `subtour_constraints.csv` | TSP subtour elimination + assignment pairs |
| `independent_set_constraints.csv` | Edge constraints `x_i * x_j == 0` |
| `qubos.csv` | Random QUBO matrices indexed by `(n_x, qubo_idx)` |

## CSV format

All constraint CSVs use a semicolon-delimited format:

```
n_vars; ['constraint_string_1', 'constraint_string_2', ...]
```

Variables are zero-indexed (`x_0`, `x_1`, ...). Read with:

```python
from analyze_results.results_helper import read_typed_csv
rows = read_typed_csv('data/knapsack_constraints.csv')
# rows: list of (n_vars, [constraint_str, ...]) tuples
```

## Constraint families

| Family | Type | Structural handler |
|---|---|---|
| `cardinality` (equality) | `sum x_i == k` | `DickeStatePrep` (exact `\|D^n_k⟩`) + XY mixer |
| `cardinality` (LEQ) | `sum x_i <= k` | `CardinalityLeqStatePrep` (superposition of Dicke states) + Grover mixer |
| `cardinality` (GEQ) | `sum x_i >= k` | `VCG` (structural Dicke-superposition support planned) |
| `knapsack` | `sum a_i x_i <= b` | `VCG` |
| `quadratic_knapsack` | `sum Q_ij x_i x_j <= b` | `VCG` |
| `flow` | Signed sum equality | `DickeStatePrep` (signed) + XY mixer |
| `assignment` | Multiple equalities | `DickeStatePrep` |
| `subtour` | TSP constraints | `VCG` |
| `independent_set` | Quadratic equality | `VCG` |

Only `knapsack` and `quadratic_knapsack` (n ≥ 3) are used for VCG training
in `run/create_vcg_database.py`.

## Regenerating data

```bash
# All constraint families:
python data/make_constraints.py

# One family only:
python data/make_constraints.py --type knapsack
python data/make_constraints.py --type quadratic

# QUBO matrices (required for experiments):
python data/make_data.py
```
