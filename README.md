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
├── 📁 run/
│   ├── run_cardinality.py    ← VCG + HybridQAOA for cardinality constraints (∑xᵢ op b)
│   ├── run_knapsack.py       ← VCG + HybridQAOA for knapsack constraints (∑cᵢxᵢ ≤ W)
│   ├── run_quadratic.py      ← VCG + HybridQAOA for quadratic constraints
│   ├── run_flow.py           ← VCG + HybridQAOA for flow conservation constraints
│   ├── run_assignment.py     ← VCG + HybridQAOA for assignment constraints
│   ├── run_subtour.py        ← VCG + HybridQAOA for subtour elimination constraints
│   ├── run_utils.py          ← Shared helpers: read_typed_csv, collect_vcg_data, collect_hybrid_data
│   └── results_handler.py    ← ResultsCollector: incremental result accumulation and persistence
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
│   └── qubos.csv
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
from run.results_handler import ResultsCollector
from run.run_utils import collect_vcg_data

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

# HybridQAOA vs PenaltyQAOA on a 3-variable cardinality-constrained QUBO
python examples/example_hybrid.py
```

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

### `run_utils.py`

```python
from run.run_utils import read_typed_csv, collect_vcg_data, collect_hybrid_data

# Parse constraint CSV (format: "n_vars; ['constraint_string']")
rows = read_typed_csv("data/cardinality_constraints.csv")

# Collect metrics after training a VCG
row = collect_vcg_data(gadget, constraint_type="cardinality")

# Collect metrics after running HybridQAOA
row = collect_hybrid_data(constraints, hybrid, qubo_string,
                          min_val=min_val, constraint_type="cardinality")
```

### `results_handler.py`

```python
from run.results_handler import ResultsCollector

collector = ResultsCollector()
collector.load("results/my_run.pkl")   # resume from existing
collector.add(row_dict)                # append one experiment row
collector.save("results/my_run.pkl")   # persist to pickle
df = collector.to_dataframe()          # pandas DataFrame
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
| `constraints` | List of constraint strings |
| `n_x` | Number of decision variables |
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
