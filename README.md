# Constraint Gadget QAOA

Code for studying constraint-handling strategies in the Quantum Approximate Optimisation Algorithm (QAOA) for solving Combinatorial Optimisation Problems (COPs).

The central idea is a **Variational Constraint Gadget (VCG)**: rather than penalising constraint violations in the cost Hamiltonian, a small QAOA circuit is built whose ground state is the uniform superposition over feasible bitstrings. This gadget then acts as a structured state-preparation oracle and mixer inside the problem QAOA, biasing the search toward the feasible subspace from the outset.

## File Structure

```
📦 constraint_gadgets/
│
├── 📄 README.md
│
├── 📁 core/
│   ├── qaoa_base.py          ← Shared QAOA logic: Hamiltonians, circuits, optimisation, resources
│   ├── constraint_handler.py ← Parsing, classification, partitioning, feasibility checking
│   ├── vcg_no_flag.py        ← Variational Constraint Gadget (VCGNoFlag) -- no ancilla qubits
│   ├── hybrid_qaoa.py        ← Hybrid QAOA: structural (VCG/Dicke) + penalty constraints
│   ├── penalty_qaoa.py       ← Standard penalty-based QAOA baseline
│   └── dicke_state_prep.py   ← Log-depth Dicke state prep + XY mixer
│
│
├── 📁 run/
│   ├── add_to_vcg_database.py        ← Train a single VCGNoFlag and register it in the gadget DB
│   ├── create_noflag_database.py     ← Populate the full gadget DB (knapsack + quadratic-knapsack)
│   ├── generate_experiment_params.py ← Enumerate HybridQAOA vs PenaltyQAOA tasks → JSONL
│   ├── run_hybrid_vs_penalty.py      ← Run the experiment sweep; stores optimal_x for P(opt)
│   └── params/
│       ├── experiment_params.jsonl   ← 500 generated experiment tasks
│       └── vcg_params.jsonl          ← VCG training task list
│
├── 📁 analyze_results/       ← Analysis and plotting package
│   ├── __init__.py
│   ├── results_helper.py     ← ResultsCollector, GadgetDatabase, remap helpers, collect_vcg/hybrid/penalty_data
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
│   ├── example_vcg.py        ← VCGNoFlag demo: train on a single constraint, plot counts
│   ├── example_hybrid.py     ← HybridQAOA vs PenaltyQAOA on a three-constraint QUBO
│   ├── results/              ← Saved result pickles (e.g. vcg_layer_sweep.pkl)
│   └── figures/              ← Generated plots (AR, timing, distributions)
│
├── 📁 slurm/  (HPC)
│   ├── submit_all.sh         ← Full pipeline: generate params + submit all jobs in dependency order
│   ├── vcg_array.sh          ← SLURM array: train one VCG per task
│   ├── vcg_merge.sh          ← Single-node: merge VCG pickles → gadgets/gadget_db.pkl
│   ├── experiment_array.sh   ← SLURM array: run one experiment task per job
│   └── experiment_merge.sh   ← Single-node: merge results → results/hybrid_vs_penalty.pkl
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
from core.vcg_no_flag import VCGNoFlag

gadget = VCGNoFlag(
    constraints=["x_0 + x_1 + x_2 == 1"],
    ar_threshold=0.999,
    max_layers=8,
    qaoa_restarts=5,
    qaoa_steps=150,
    ma_restarts=20,
    ma_steps=200,
    lr=0.05,
    samples=10_000,
)
gadget.train(verbose=True)
counts = gadget.do_counts_circuit(shots=10_000)
p_feas = gadget.p_feasible(shots=10_000)
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
from analyze_results.results_helper import (
    ResultsCollector, GadgetDatabase,
    collect_vcg_data, collect_hybrid_data, collect_penalty_data,
)

# Full results (all metrics) – for analysis
collector = ResultsCollector()
collector.load("results/cardinality_constraint_results.pkl")  # resume if exists

# Gadget database (minimal fields only) – for HybridQAOA lookup
# collect_vcg_data registers the gadget automatically when gadget_db_path is given
row = collect_vcg_data(gadget, constraint_type="cardinality",
                       gadget_db_path="gadgets/gadget_db.pkl")
collector.add(row)
collector.save("results/cardinality_constraint_results.pkl")

df = collector.to_dataframe()
```

The gadget database stores only the 6 fields required by HybridQAOA
(`constraints`, `n_layers`, `angle_strategy`, `outcomes`, `Hamiltonian`, `opt_angles`),
keeping it lean relative to the full results file. Entries are deduplicated automatically.

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
| B | `6*x_3 + 2*x_4 + 2*x_5 <= 3` | {3, 4, 5} | Structural – VCGNoFlag gadget |
| C | `x_1 + x_4 + x_6 <= 1` | {1, 4, 6} | Penalized (overlaps A and B) |

Constraints A and B are **disjoint** (no shared variables), while C deliberately overlaps both groups
(x_1 ∈ A, x_4 ∈ B, x_6 is free).

**Qubit layout – 8 qubits total**

| Wires | Count | Role |
|---|---|---|
| 0–6 | 7 | Decision variables x_0 … x_6 |
| 7 | 1 | Slack qubit for constraint C (`x_1+x_4+x_6 + s = 1`, s ∈ {0,1}) |

Constraint A (Dicke) and constraint B (VCGNoFlag) use no ancilla qubits — both operate directly
on the decision-variable wires.
Constraint C's inequality `<= 1` needs one binary slack qubit because the minimum feasible LHS value
is 0 and the RHS is 1, so `n_slack = ceil(1 − 0) = 1`.

**Step 2 – Route constraints by type**

`constraint_handler.is_dicke_compatible` classifies each parsed constraint:

- **Dicke-compatible** (A): all coefficients are +1, equality operator, integer RHS.
  HybridQAOA prepares the uniform superposition over feasible assignments exactly using a log-depth
  W-state circuit and an XY mixer – no flag qubit, zero approximation error.

- **Not Dicke-compatible** (B): non-unit coefficients or inequality operator.
  HybridQAOA trains a flag-free VCGNoFlag gadget whose ground state is the uniform
  superposition over feasible assignments for B, then embeds it as the initial state and uses a
  Grover mixer.  P(feasible) is measured by directly evaluating the constraint on bitstrings —
  no ancilla qubit is involved.

- **Penalized** (C): constraint spans variables from both groups, so it cannot be folded into either
  structural circuit cleanly.  It is instead converted to a quadratic penalty term
  δ·(x_1 + x_4 + x_6 − 1 + s)² and added to the cost Hamiltonian.

**Step 3 – Solve with HybridQAOA**

```python
hybrid = HybridQAOA(
    qubo=Q,                         # 7x7 QUBO loaded from data/qubos.csv
    all_constraints=parsed,         # [A, B, C]
    structural_indices=[0, 1],      # A (Dicke) + B (VCGNoFlag) enforced structurally
    penalty_indices=[2],            # C penalized
    penalty_str=[delta],            # penalty weights for penalized constraints
    penalty_pen=delta,              # cost-Hamiltonian penalty weight
    angle_strategy='ma-QAOA',
    mixer='Grover',                 # reflects about the composed A+B state
    n_layers=1,
    steps=50,
    num_restarts=10,
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
- `hybrid_example_counts.png` – top-20 outcome distributions with 5-category colour coding:

| Colour | Meaning |
|---|---|
| foam | Optimal |
| pine | All feasible |
| gold | Structural ✓, Penalty ✗ |
| rose | Structural ✗, Penalty ✓ |
| love | All infeasible |

## How the VCG Works

A **Variational Constraint Gadget (VCGNoFlag)** is a small QAOA circuit whose
ground state is the uniform superposition over all bitstrings that satisfy a
given constraint.  Once trained, it acts as both the initial state and the
Grover mixer inside HybridQAOA, keeping the search within the feasible
subspace.  The Hamiltonian is defined directly on the decision-variable qubits
— no ancilla or flag qubit is used.

### Step 1 — Constraint Hamiltonian

VCGNoFlag builds a diagonal Hamiltonian whose eigenvalues encode feasibility:

```
H_constraint = diag(outcomes)   where outcomes[s] = -1 if s is feasible, +1 otherwise
```

Concretely:

1. **Truth table** — enumerate every assignment of the `n_x` decision variables.
   For each assignment, evaluate whether all constraints are satisfied.
   All `2^n_x` states are labelled −1 (feasible) or +1 (infeasible).

2. **Pauli decomposition** — because the Hamiltonian is diagonal, all its
   Pauli terms are products of Z operators (no off-diagonal terms).  There
   are at most `2^n` such Z-string terms.  Their coefficients are computed
   via a **Walsh-Hadamard transform (WHT)** of the `outcomes` vector:

   ```
   c_S = (1/2^n) · Σ_x  outcomes[x] · (−1)^{popcount(x & S)}
   ```

   WHT runs in O(n · 2^n) time and O(2^n) memory.  This replaces the naive
   approach of constructing a full `2^n × 2^n` matrix and calling
   `qml.pauli_decompose`, which costs O(4^n) in both time and memory and
   fails for constraints beyond n≈13 qubits due to OOM.

   The result — `num_gamma` non-trivial Pauli terms — determines the number
   of independent cost angles for ma-QAOA.

> **The `decompose` flag.**  Because the VCG Hamiltonian is always diagonal,
> all its Pauli terms are products of Z operators and therefore mutually
> commute.  The decomposed product `∏_k exp(−iγ w_k P_k)` and the
> matrix-exponential form `exp(−iγ H)` implement **exactly the same unitary**
> for standard QAOA.  `decompose=True` is nonetheless always recommended
> because:
> - it is **required** for ma-QAOA (each Pauli term needs its own angle), and
> - it keeps the circuit in native gate form (MultiRZ), enabling exact
>   parameter-shift gradients and transparent resource counting.

### Step 2 — QAOA circuit

```
|+⟩^n  →  [Cost(γ) · Mixer(β)]^p  →  measure
```

- **Initialisation**: Hadamard on every qubit → equal superposition.
- **Cost layer**: for each non-identity Pauli term `k`,
  apply `MultiRZ(w_k · γ_k, wires)`.
- **Mixer layer**: `RX(β_i, wire_i)` on every qubit (standard X-mixer).
- Repeat for `p = n_layers` rounds.

### Step 3 — Angle strategies

| Strategy | Parameters per layer | Description |
|---|---|---|
| `QAOA` | 2 (one γ, one β) | All Pauli terms share γ; all qubits share β |
| `ma-QAOA` | `num_gamma + num_beta` | Independent angle per Pauli term and per qubit |

QAOA is a special case of ma-QAOA (all γ equal, all β equal), so ma-QAOA's
optimal AR is always ≥ QAOA's.  In practice, QAOA has a **structural
ceiling** below AR=1 for constraints with many Pauli terms — the shared γ
cannot independently weight each term.  For a 5-variable knapsack, QAOA
saturates at AR≈0.985 regardless of depth.

### Step 4 — Depth sweep strategy

A single QAOA run at p=1 (2 parameters, ~8 s) provides a warm-start for
ma-QAOA.  Its optimal angles are broadcast (one γ → all `num_gamma` entries,
one β → all `num_beta` entries) as the first restart's starting point:

```python
# Fast QAOA p=1 warm-up
opt_cost, qaoa_angles = qaoa_gadget.optimize_angles(
    qaoa_gadget.do_evolution_circuit,
)

# ma-QAOA p=1: first restart seeded from QAOA
opt_cost, _ = ma_gadget.optimize_angles(
    ma_gadget.do_evolution_circuit,
    starting_angles_from_qaoa=qaoa_angles,
)

# ma-QAOA p>1: joint re-opt all layers, warm-started from previous depth
opt_cost, _ = ma_gadget.optimize_angles(
    ma_gadget.do_evolution_circuit,
    prev_layer_angles=prev_best_ma,
)
```

### Step 5 — Quality metric

```
AR = (⟨H_constraint⟩ − C_max) / (C_min − C_max)
```

For a binary VCG, `C_min = −1` (all weight on good states) and
`C_max = +1` (all weight on bad states), so `AR = (⟨H⟩ − 1) / −2`.
A gadget is considered well-trained when `AR ≥ 0.95`.

## Running Experiments

### Build the VCG gadget database

```bash
# Train all knapsack / quadratic-knapsack VCGNoFlag gadgets sequentially
python run/create_noflag_database.py --db gadgets/gadget_db.pkl

# Or add a single constraint
python run/add_to_vcg_database.py \
    --constraints "6*x_0 + 2*x_1 + 2*x_2 <= 3" \
    --db gadgets/gadget_db.pkl
```

### Generate and run HybridQAOA vs PenaltyQAOA experiments

```bash
# Enumerate experiment parameter combinations
python run/generate_experiment_params.py \
    --output run/params/experiment_params.jsonl --max-tasks 500

# Run all experiments sequentially
python run/run_hybrid_vs_penalty.py \
    --params run/params/experiment_params.jsonl \
    --db gadgets/gadget_db.pkl
```

### Submitting SLURM array jobs

```bash
# Step 1 – generate task lists
python slurm/generate_params.py

# Step 2 – submit array jobs (adapt .sh template to your cluster)
#   VCG training: --array=0-<N_vcg-1>
#     python run/create_noflag_database.py --task-id $SLURM_ARRAY_TASK_ID
#   Experiments:  --array=0-<N_exp-1>
#     python run/run_hybrid_vs_penalty.py --task-id $SLURM_ARRAY_TASK_ID

# Step 3 – merge per-task results
python run/create_noflag_database.py --merge --db gadgets/gadget_db.pkl
python run/run_hybrid_vs_penalty.py --merge --output results/hybrid_vs_penalty.pkl
```

## Core API

### `analyze_results/results_helper.py`

```python
from analyze_results.results_helper import (
    ResultsCollector, GadgetDatabase,
    read_typed_csv, remap_constraint_to_vars, remap_to_zero_indexed,
    collect_vcg_data, collect_hybrid_data, collect_penalty_data,
)

# Parse constraint CSV (format: "n_vars; ['constraint_string']")
rows = read_typed_csv("data/cardinality_constraints.csv")

# Embed a zero-indexed constraint onto arbitrary QUBO variables
c = remap_constraint_to_vars("x_0 + x_1 == 1", [3, 5])  # → 'x_3 + x_5 == 1'

# Train VCG, collect full metrics, and register the gadget in the database
row_vcg = collect_vcg_data(gadget, constraint_type="cardinality",
                           gadget_db_path="gadgets/gadget_db.pkl")

row_hybrid  = collect_hybrid_data(constraints, hybrid, qubo_string, min_val=min_val)
row_penalty = collect_penalty_data(constraints, penalty_solver, qubo_string, min_val=min_val)

# Accumulate full results and persist to pickle
collector = ResultsCollector()
collector.load("results/my_run.pkl")   # resume from existing
collector.add(row_hybrid)
collector.add(row_penalty)
collector.save("results/my_run.pkl")
df = collector.to_dataframe()
```

### VCGNoFlag Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `constraints` | list[str] | — | Constraint strings, e.g. `["x_0 + x_1 == 1"]` |
| `ar_threshold` | float | `0.999` | Stop training when AR >= this value |
| `entropy_threshold` | float | `0.9` | Stop when normalised entropy >= this value (once AR met) |
| `max_layers` | int | `8` | Maximum ma-QAOA layers in the sweep |
| `qaoa_restarts` | int | `5` | Random restarts for Stage 1 QAOA warm-start |
| `qaoa_steps` | int | `150` | Optimisation steps for Stage 1 QAOA warm-start |
| `ma_restarts` | int | `20` | Random restarts per ma-QAOA layer |
| `ma_steps` | int | `200` | Optimisation steps per ma-QAOA layer |
| `lr` | float | `0.05` | Adam learning rate |
| `samples` | int | `10_000` | Measurement shots for counts / P(feasible) |
| `decompose` | bool | `True` | Decompose Hamiltonian into Pauli terms (required for ma-QAOA) |

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
| `penalty_str` | list[float] | `None` | Penalty weights for penalized constraints |
| `steps` | int | `50` | Optimisation steps per restart |
| `num_restarts` | int | `5` | Random restarts per layer |
| `cqaoa_steps` | int | `30` | Steps for inline VCG training when gadget not in DB |
| `cqaoa_num_restarts` | int | `5` | Restarts for inline VCG training |
| `pre_made` | bool | `False` | Load pre-trained VCG angles from `gadget_path` |

## Constraint Families

| Family | Example constraint | CSV file |
|---|---|---|
| Cardinality (equality) | `x_0 + x_1 + x_2 == 1` | `cardinality_constraints.csv` |
| Cardinality (LEQ) | `x_0 + x_1 + x_2 <= 2` | `cardinality_constraints.csv` |
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
| `min_val` | Optimal feasible QUBO value (brute force) |
| `optimal_x` | List of optimal feasible bitstrings (brute force; used to compute P(opt)) |
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
