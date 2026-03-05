# Constraint Gadget QAOA

Code for studying constraint-handling strategies in the Quantum Approximate Optimisation Algorithm (QAOA) for solving Combinatorial Optimisation Problems (COPs).

The central idea is a **Variable Constraint Gadget (VCG)**: rather than penalising constraint violations in the cost Hamiltonian, a small QAOA circuit is built whose ground state is the uniform superposition over feasible bitstrings. This gadget then acts as a structured state-preparation oracle and mixer inside the problem QAOA, biasing the search toward the feasible subspace from the outset.

## File Structure

```
📦 constraint_gadgets/
│
├── 📄 README.md
│
├── 📁 core/
│   ├── qaoa_base.py          ← Shared QAOA logic: Hamiltonians, circuits, optimisation, resources
│   ├── constraint_handler.py ← Parsing, classification, partitioning, feasibility checking
│   ├── vcg.py                ← Variable Constraint Gadget (VCG) QAOA
│   ├── hybrid_qaoa.py        ← Hybrid QAOA: structural (VCG/Dicke) + penalty constraints
│   ├── penalty_qaoa.py       ← Standard penalty-based QAOA baseline
│   └── dicke_state_prep.py   ← Log-depth Dicke state prep + XY mixer
│
│   ├── results_helper.py     ← ResultsCollector, read_typed_csv, remap helpers, collect_vcg/hybrid/penalty_data
│
├── 📁 run/
│   ├── run_cardinality.py    ← VCG + HybridQAOA for cardinality constraints (∑xᵢ op b)
│   ├── run_knapsack.py       ← VCG + HybridQAOA for knapsack constraints (∑cᵢxᵢ ≤ W)
│   ├── run_quadratic.py      ← VCG + HybridQAOA for quadratic constraints
│   ├── run_flow.py           ← VCG + HybridQAOA for flow conservation constraints
│   ├── run_assignment.py     ← VCG + HybridQAOA for assignment constraints
│   └── run_subtour.py        ← VCG + HybridQAOA for subtour elimination constraints
│
├── 📁 analyze_results/       ← Analysis and plotting package
│   ├── __init__.py
│   ├── data_loader.py        ← Load/filter/clean results DataFrames
│   ├── metrics.py            ← P(feasible), P(optimal), AR augmentation, summary stats
│   ├── plot_utils.py         ← Shared matplotlib styling (rose-pine palette)
│   ├── plot_ar.py            ← Approximation ratio plots
│   ├── plot_feasibility.py   ← P(feasible) and P(optimal) plots
│   ├── plot_resources.py     ← Circuit depth, shot budget, time breakdown
│   ├── statistical_tests.py  ← Mann-Whitney U, Kruskal-Wallis significance tests
│   ├── main_analysis.py      ← CLI entry point for the full analysis pipeline
│   └── README.md
│
├── 📁 examples/
│   ├── example_vcg.py        ← VCG demo: train on a single constraint, plot counts
│   └── example_hybrid.py     ← HybridQAOA vs PenaltyQAOA on a cardinality-constrained QUBO
│
├── 📁 slurm/  (HPC)
│   ├── generate_params.py    ← Regenerate all param files declaratively
│   ├── vcg_params.txt        ← VCG sweep: constraint families × n_layers
│   └── hybrid_params.txt     ← HybridQAOA sweep: constraint families × n_layers
│
├── 📁 data/                  ← Constraint CSVs, QUBO instances, and data utilities
│   ├── make_data.py              ← QUBO generation and optimal-x brute force search
│   ├── make_constraints.py       ← Constraint CSV generation scripts
│   ├── cardinality_constraints.csv
│   ├── knapsack_constraints.csv
│   ├── quadratic_knapsack_constraints.csv
│   ├── flow_constraints.csv
│   ├── assignment_constraints.csv
│   ├── subtour_constraints.csv
│   └── qubos.csv             ← Random QUBOs, sizes 2–10 (10 per size; max = 2 × max constraint support)
│
├── 📁 results/               ← Collected experiment results (.pkl) (gitignored)
└── 📁 analysis_output/       ← Figures, stats, summaries from analysis pipeline (gitignored)
```

## Quick Start

### Build and optimise a constraint gadget (Python)

```python
from core.vcg import VCG

gadget = VCG(
    constraints=["x_0 + x_1 + x_2 == 1"],
    flag_wires=[3],
    angle_strategy="ma-QAOA",
    n_layers=1,
    steps=50,
    num_restarts=50,
)
opt_cost, opt_angles = gadget.optimize_angles(gadget.do_evolution_circuit)
counts = gadget.do_counts_circuit(shots=10_000)
```

### Solve a constrained QUBO with HybridQAOA

```python
import numpy as np
from core import constraint_handler as ch
from core.hybrid_qaoa import HybridQAOA

Q = np.array([[1, -2, 0], [-2, 3, -1], [0, -1, 2]], dtype=float)
constraints = ["x_0 + x_1 + x_2 == 1"]
parsed = ch.parse_constraints(constraints)

solver = HybridQAOA(
    qubo=Q,
    all_constraints=parsed,
    structural_indices=[0],   # enforce via Dicke state prep
    penalty_indices=[],
    angle_strategy="ma-QAOA",
    mixer="Grover",
    n_layers=1,
    steps=50,
    num_restarts=10,
)
opt_cost, counts, opt_angles = solver.solve()
```

### Collect results incrementally

```python
from analyze_results.results_helper import ResultsCollector, collect_vcg_data, collect_hybrid_data, collect_penalty_data

collector = ResultsCollector()
collector.load("results/cardinality_constraint_results.pkl")  # resume if exists

row = collect_vcg_data(gadget, constraint_type="cardinality")
collector.add(row)
collector.save("results/cardinality_constraint_results.pkl")

df = collector.to_dataframe()
```

### Run the toy examples

```bash
# VCG demo: train on x_0 + x_1 + x_2 == 1, print AR / P(feasible), plot counts
python examples/example_vcg.py

# HybridQAOA vs PenaltyQAOA – three-constraint COP on 7 decision variables
python examples/example_hybrid.py
```

#### How `example_hybrid.py` works

The example builds a three-constraint combinatorial optimisation problem on 7 binary decision
variables (`x_0 … x_6`) and compares HybridQAOA against a full-penalisation baseline.

**Step 1 – Load constraints from data/**

Two CSV files are read (`data/cardinality_constraints.csv` and `data/knapsack_constraints.csv`).
Three constraints are selected and embedded onto specific variable subsets using
`remap_constraint_to_vars`:

| Label | Constraint | Variables | Handling |
|---|---|---|---|
| A | `x_0 + x_1 + x_2 == 1` | {0, 1, 2} | Structural – Dicke state prep |
| B | `6*x_3 + 2*x_4 + 2*x_5 <= 3` | {3, 4, 5} | Structural – VCG gadget |
| C | `x_1 + x_4 + x_6 <= 1` | {1, 4, 6} | Penalized (overlaps A and B) |

Constraints A and B are **disjoint** (no shared variables), while C deliberately overlaps both groups
(x_1 ∈ A, x_4 ∈ B, x_6 is free).

**Step 2 – Route constraints by type**

`constraint_handler.is_dicke_compatible` classifies each parsed constraint:

- **Dicke-compatible** (A): all coefficients are +1, equality operator, integer RHS.
  HybridQAOA prepares the uniform superposition over feasible assignments exactly using a log-depth
  W-state circuit and an XY mixer – no flag qubit, zero approximation error.

- **Not Dicke-compatible** (B): non-unit coefficients or inequality operator.
  HybridQAOA trains a Variable Constraint Gadget (VCG) whose ground state is the uniform
  superposition over feasible assignments for B, then embeds it as the initial state and uses a
  Grover mixer with one flag qubit marking (un)satisfying assignments.

- **Penalized** (C): constraint spans variables from both groups, so it cannot be folded into either
  structural circuit cleanly.  It is instead converted to a quadratic penalty term
  δ·(x_1 + x_4 + x_6 − 1 + s)² and added to the cost Hamiltonian.

**Step 3 – Solve with HybridQAOA**

```python
hybrid = HybridQAOA(
    qubo=Q,                         # 7×7 QUBO loaded from data/qubos.csv
    all_constraints=parsed,         # [A, B, C]
    structural_indices=[0, 1],      # A (Dicke) + B (VCG) enforced structurally
    penalty_indices=[2],            # C penalized
    penalty_str=[delta],            # flag-qubit penalty weight
    penalty_pen=delta,              # cost-Hamiltonian penalty weight
    angle_strategy='ma-QAOA',
    mixer='Grover',                 # reflects about the composed A+B state
    n_layers=1,
    steps=50,
    num_restarts=10,
    pre_made=False,                 # train the VCG for B from scratch
)
opt_cost, counts, opt_angles = hybrid.solve()
```

The Grover mixer reflects about the state prepared by the **composed** A+B circuit, so the search
stays within the joint feasible subspace of A and B throughout optimisation.

**Step 4 – Baseline: PenaltyQAOA**

All three constraints are converted to penalty terms and added to the Hamiltonian.  The circuit
starts from |+⟩^n with no structured state preparation, providing a direct comparison.

**Step 5 – Analyse and plot**

Metrics are computed over 10 000 measurement shots (auxiliary bits stripped):

- **AR** (Approximation Ratio): `(⟨H⟩ − C_max) / (C_min − C_max)`
- **P(feasible)**: fraction of samples satisfying all three constraints
- **P(optimal)**: fraction of samples achieving the brute-force optimal QUBO value

Two figures are saved to `examples/figures/`:

- `hybrid_example_metrics.png` – side-by-side bar chart of AR, P(feasible), P(optimal)
- `hybrid_example_counts.png` – top-20 outcome distributions coloured by feasibility/optimality

## Running Experiments

### Single constraint family (command line)

```bash
# Train VCG gadgets for cardinality constraints up to n=5
python run/run_cardinality.py --corp constraint --max_n 5

# Run HybridQAOA on cardinality constraints, 2 QAOA layers
python run/run_cardinality.py --corp hybrid --max_n 5 --n_layers 2

# Other constraint families follow the same interface
python run/run_knapsack.py   --corp constraint --max_n 5
python run/run_flow.py       --corp hybrid     --max_n 5 --n_layers 1
python run/run_assignment.py --corp constraint --max_n 4
python run/run_subtour.py    --corp hybrid     --max_n 4 --n_layers 1
```

### Analyse results

```bash
python analyze_results/main_analysis.py \
    --vcg   results/cardinality_constraint_results.pkl \
    --hybrid results/hybrid_cardinality_results.pkl \
    --output-dir analysis_output/
```

Output directories:
- `analysis_output/figures/ar/`
- `analysis_output/figures/feasibility/`
- `analysis_output/figures/resources/`
- `analysis_output/summaries/`
- `analysis_output/statistical_tests/`

### Submitting HPC jobs

```bash
# Regenerate param files after changing the parameter space
python slurm/generate_params.py

# Submit (adapt the .sh template to your cluster scheduler)
# Each line of vcg_params.txt / hybrid_params.txt is one array task:
#   constraint_type  corp  n_layers
#   cardinality      constraint  1
#   knapsack         hybrid      2
```

## Core API

### `analyze_results/results_helper.py`

```python
from analyze_results.results_helper import (
    ResultsCollector,
    read_typed_csv, remap_constraint_to_vars, remap_to_zero_indexed,
    collect_vcg_data, collect_hybrid_data, collect_penalty_data,
)

# Parse constraint CSV (format: "n_vars; ['constraint_string']")
rows = read_typed_csv("data/cardinality_constraints.csv")

# Embed a zero-indexed constraint onto arbitrary QUBO variables
c = remap_constraint_to_vars("x_0 + x_1 == 1", [3, 5])  # → 'x_3 + x_5 == 1'

# Collect metrics (train + sample) and return a result row dict
row_vcg     = collect_vcg_data(gadget, constraint_type="cardinality")
row_hybrid  = collect_hybrid_data(constraints, hybrid, qubo_string, min_val=min_val)
row_penalty = collect_penalty_data(constraints, penalty_solver, qubo_string, min_val=min_val)

# Accumulate rows and persist to pickle
collector = ResultsCollector()
collector.load("results/my_run.pkl")   # resume from existing
collector.add(row_hybrid)
collector.add(row_penalty)
collector.save("results/my_run.pkl")
df = collector.to_dataframe()
```

### VCG Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `constraints` | list[str] | — | Constraint strings, e.g. `["x_0 + x_1 == 1"]` |
| `flag_wires` | list[int] | — | Flag qubit wire indices (one per constraint) |
| `angle_strategy` | str | `"ma-QAOA"` | `"QAOA"` or `"ma-QAOA"` |
| `n_layers` | int | `1` | QAOA circuit depth |
| `decompose` | bool | `True` | Decompose Hamiltonian into Pauli terms |
| `single_flag` | bool | `False` | Use a single shared flag qubit |
| `steps` | int | `50` | Optimisation steps per restart |
| `num_restarts` | int | `100` | Random restarts |
| `learning_rate` | float | `0.01` | Adam step size |

### HybridQAOA Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `qubo` | np.ndarray | — | QUBO matrix (n_x × n_x) |
| `all_constraints` | list[ParsedConstraint] | — | All parsed constraints |
| `structural_indices` | list[int] | — | Indices enforced via gadget / Dicke prep |
| `penalty_indices` | list[int] | — | Indices enforced via penalty term |
| `angle_strategy` | str | `"ma-QAOA"` | `"QAOA"` or `"ma-QAOA"` |
| `mixer` | str | `"Grover"` | `"Grover"`, `"X-Mixer"`, or `"XY"` |
| `n_layers` | int | `1` | QAOA circuit depth |
| `penalty_str` | list[float] | `None` | Flag-qubit penalty weights |
| `steps` | int | `50` | Optimisation steps per restart |
| `num_restarts` | int | `10` | Random restarts |
| `pre_made` | bool | `False` | Load pre-trained VCG angles from `gadget_path` |

## Constraint Families

| Family | Example constraint | CSV file |
|---|---|---|
| Cardinality | `x_0 + x_1 + x_2 == 1` | `cardinality_constraints.csv` |
| Knapsack | `3*x_0 + 2*x_1 + x_2 <= 4` | `knapsack_constraints.csv` |
| Quadratic knapsack | `x_0*x_1 + 2*x_2 <= 2` | `quadratic_knapsack_constraints.csv` |
| Flow conservation | `x_0 + x_1 - x_2 - x_3 == 0` | `flow_constraints.csv` |
| Assignment | `x_0 + x_1 == 1` (rows + cols) | `assignment_constraints.csv` |
| Subtour elimination | Multi-constraint TSP subtours | `subtour_constraints.csv` |

## Angle Strategies

- **QAOA** — one shared γ and β per layer.
- **ma-QAOA** — one independent angle per Pauli term and per qubit per layer.

## Mixers

| Mixer | Description |
|---|---|
| **Grover** | Reflects about the gadget-prepared feasible state |
| **X-Mixer** | Standard transverse-field mixer on all qubits |
| **XY / Ring-XY** | Hamming-weight-preserving mixer for Dicke-enforced constraints |

## Output Format

`collect_vcg_data` and `collect_hybrid_data` return a flat dict that `ResultsCollector` accumulates into a DataFrame. Key columns:

| Column | Description |
|---|---|
| `constraint_type` | Constraint family label (e.g. `"cardinality"`) |
| `constraints` | Constraint strings remapped to their QUBO variable positions |
| `var_assignment` | List of QUBO variable indices the constraint acts on |
| `n_x` | Total number of QUBO variables (>= constraint support) |
| `n_c` | Number of constraints |
| `angle_strategy` | `"QAOA"` or `"ma-QAOA"` |
| `n_layers` | Number of QAOA layers |
| `opt_angles` | Optimised angle array |
| `opt_cost` | Final expectation value ⟨H⟩ |
| `AR` | Approximation ratio (opt_cost − C_max) / (C_min − C_max) |
| `counts` | Measurement outcome distribution dict |
| `est_shots` | Estimated shot budget from Pauli grouping |
| `resources` | PennyLane gate resource object (VCG only) |
| `hamiltonian_time` | Wall time for Hamiltonian construction (s) |
| `optimize_time` | Wall time for angle optimisation (s) |
| `counts_time` | Wall time for sampling (s) |
| `min_val` | Optimal feasible QUBO value (hybrid only) |
| `mixer` | Mixer used (hybrid only) |

## Dependencies

```
pennylane >= 0.38
pennylane-lightning
numpy
pandas
matplotlib
seaborn
scipy
```

Install with:

```bash
pip install pennylane pennylane-lightning numpy pandas matplotlib seaborn scipy
```

## Links

- [Analysis Package](analyze_results/README.md)
