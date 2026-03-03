# Constraint Gadget QAOA

Code for studying constraint-handling strategies in the Quantum Approximate Optimisation Algorithm (QAOA) for solving Quadratically-Constrained Binary Optimisation (QCBO) problems.

The central idea is a **constraint gadget**: rather than penalising constraint violations in the cost Hamiltonian, we build a small QAOA circuit whose ground state is the uniform superposition over feasible bitstrings. This gadget is then used as a structured state-preparation oracle and mixer inside the problem QAOA, biasing the search toward the feasible subspace from the outset.

---

## Methods

| Class | File | Description |
|---|---|---|
| `ConstraintQAOA` | `constraint_qaoa.py` | Builds a constraint Hamiltonian from a truth table (feasible states → −1, infeasible → +1) and optimises QAOA angles to prepare the feasible subspace. Acts as a structured mixer for `ProblemQAOA`. |
| `ProblemQAOA` | `problem_qaoa.py` | Solves a QCBO using one or more `ConstraintQAOA` gadgets for state preparation and a Grover or X mixer. |
| `PenaltyQAOA` | `penalty_qaoa.py` | Standard penalty-based QAOA: adds δ·(constraint)² terms to the QUBO and introduces slack variables for inequalities. Baseline comparison. |
| `HybridQAOA` | `hybrid_qaoa.py` | Partitions constraints into *structural* (enforced via gadgets or Dicke circuits) and *penalised* groups. Balances expressibility and feasibility bias. |
| `DickeStatePrep` | `dicke_state_prep.py` | Log-depth preparation of Dicke states for cardinality constraints (∑xᵢ = b) with an XY or Ring-XY mixer. |

### Shared utilities

| Module | Description |
|---|---|
| `qaoa_base.py` | QUBO↔Ising conversion, Hamiltonian builders, cost unitary / mixer circuit blocks, angle initialisation and splitting, Adam optimisation loop, resource estimation, Pauli expansion helpers for penalty terms. |
| `constraint_handler.py` | Constraint parsing, classification (Dicke / weighted-sum / quadratic), disjointness analysis, structural vs. penalty partitioning, slack variable allocation, and feasibility checking. |

### Angle strategies

- **QAOA** — one shared γ and β per layer.
- **ma-QAOA** — one independent angle per Pauli term and per qubit per layer.

### Mixers

| Mixer | Use case |
|---|---|
| Grover | Reflects about the gadget-prepared feasible state; default for `ProblemQAOA`. |
| X-Mixer | Standard QAOA mixer on all qubits; default for `PenaltyQAOA`. |
| XY / Ring-XY | Hamming-weight-preserving mixer for Dicke-enforced cardinality constraints. |

---

## Dependencies

```
pennylane >= 0.38
pennylane-lightning
numpy
pandas
matplotlib
seaborn
```

Install with:

```bash
pip install pennylane pennylane-lightning numpy pandas matplotlib seaborn
```

---

## Quick start

### Build and optimise a constraint gadget

```python
from constraint_qaoa import ConstraintQAOA

gadget = ConstraintQAOA(
    constraints=["x_0 + x_1 + x_2 == 2"],
    flag_wires=[3],
    angle_strategy="ma-QAOA",
    n_layers=1,
    num_restarts=50,
    steps=100,
)
opt_cost, opt_angles = gadget.optimize_angles(gadget.do_evolution_circuit)
counts = gadget.do_counts_circuit(shots=10_000)
```

### Solve a QCBO with the constraint gadget

```python
import numpy as np
from constraint_qaoa import ConstraintQAOA
from problem_qaoa import ProblemQAOA

Q = np.array([[1, -2, 0], [-2, 3, -1], [0, -1, 2]], dtype=float)

gadget = ConstraintQAOA(
    constraints=["x_0 + x_1 + x_2 == 2"],
    flag_wires=[3],
    angle_strategy="ma-QAOA",
    n_layers=1,
    pre_made=False,
)
gadget.optimize_angles(gadget.do_evolution_circuit)

solver = ProblemQAOA(
    qubo=Q,
    state_prep=[gadget],
    mixer="Grover",
    angle_strategy="ma-QAOA",
    penalty=[10.0],
    n_layers=1,
)
opt_cost, opt_angles = solver.optimize_angles(solver.do_evolution_circuit)
counts = solver.do_counts_circuit(shots=10_000)
```

### Penalty-based baseline

```python
import numpy as np
from penalty_qaoa import PenaltyQAOA

Q = np.array([[1, -2, 0], [-2, 3, -1], [0, -1, 2]], dtype=float)

solver = PenaltyQAOA(
    qubo=Q,
    constraints=["x_0 + x_1 + x_2 <= 2", "x_1 + x_2 >= 1"],
    penalty=10.0,
    angle_strategy="ma-QAOA",
    n_layers=1,
)
opt_cost, opt_angles = solver.optimize_angles(solver.do_evolution_circuit)
```

---

## Running experiments

### Single-constraint experiments

```bash
# Build constraint gadgets for ∑xᵢ == b, n ∈ [1, 5]
python run_single_constraint.py --corp constraint --op equals --max_n 5

# Solve QCBOs using pre-built gadgets
python run_single_constraint.py --corp problem --op equals --max_n 5 --n_layers 1
```

Supported operators: `equals`, `geq`, `leq`, `less`, `greater`.

### Two-constraint experiments

```bash
# Build gadgets for two overlapping constraints with a given support
python run_two_constraint.py --corp constraint --support 2

# Solve QCBOs (single flag qubit, 2 QAOA layers)
python run_two_constraint.py --corp problem --support 2 --single_flag --n_layers 2
```

`--support` controls how many variables the two constraints share (1 = minimal, 3 = maximal overlap).

### Generating problem data

```python
from make_data import generate_n_qubos, write_qubos_to_file

qubos = generate_n_qubos(num_qubo=10, min_n=2, max_n=5)
write_qubos_to_file(qubos, "qubos.csv", min_n=2, max_n=5, results_dir="./results/")
```

---

## Directory structure (planned)

The codebase will be reorganised into the following layout:

```
constraint_gadgets/
│
├── core/                   # Main classes and shared utilities
│   ├── qaoa_base.py
│   ├── constraint_handler.py
│   ├── constraint_qaoa.py
│   ├── problem_qaoa.py
│   ├── penalty_qaoa.py
│   ├── hybrid_qaoa.py
│   └── dicke_state_prep.py
│
├── run/                    # Experiment entry points
│   ├── run_single_constraint.py
│   └── run_two_constraint.py
│
├── examples/               # Self-contained toy examples and tutorials
│
├── slurm/                  # Job scripts for HPC / remote cluster runs
│
├── analyze_results/        # Result parsing and visualisation
│   ├── make_data.py
│   └── analyze_results.py
│
├── data/                   # Input problem data (QUBOs, constraint files)
│   └── qubos.csv
│
├── results/                # Output pickle DataFrames from experiments
│
├── analysis_output/        # Processed outputs
│   ├── figures/            # Publication-quality plots
│   ├── statistical_results/
│   └── summaries/
│
└── docs/                   # Documentation
```

---

## Results schema

Each experiment writes a pandas DataFrame to a `.pkl` file. Key columns:

| Column | Description |
|---|---|
| `constraints` | List of constraint strings |
| `n_x` | Number of decision variables |
| `angle_strategy` | `"QAOA"` or `"ma-QAOA"` |
| `n_layers` | Number of QAOA layers p |
| `opt_angles` | Optimised angle array |
| `opt_cost` | Final expectation value ⟨H⟩ |
| `AR` | Approximation ratio (C_opt − C_max) / (C_min − C_max) |
| `counts` | Measurement outcome distribution |
| `est_shots` | Estimated shot budget (Pauli grouping) |
| `hamiltonian_time` | Wall time for Hamiltonian construction (s) |
| `optimize_time` | Wall time for angle optimisation (s) |
