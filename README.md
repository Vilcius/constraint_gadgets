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
│   ├── vcg.py                ← Variational Constraint Gadget (VCG) -- no ancilla qubits
│   ├── pc_qaoa.py            ← PC-QAOA: structural (VCG/Dicke) + penalty constraints
│   ├── penalty_qaoa.py       ← Standard penalty-based QAOA baseline
│   ├── dicke_state_prep.py   ← Log-depth Dicke state prep + XY mixer
│   └── resource_estimation.py ← Analytical gate-count resource estimation
│
├── 📁 run/
│   ├── generate_vcg_params.py        ← Extract unique VCG constraints → vcg_params_experiments.jsonl
│   ├── generate_experiment_params.py ← Enumerate PC-QAOA vs PenaltyQAOA tasks → JSONL
│   ├── create_vcg_database.py        ← Train all VCG gadgets; save to gadgets/vcg_db.pkl
│   ├── run_pc_qaoa_vs_penalty.py     ← Run experiments (single COP or merge pending results)
│   └── params/
│       ├── vcg_params_experiments.jsonl          ← VCG training task list
│       ├── experiment_params_overlapping.jsonl   ← 250 overlapping COP tasks
│       └── experiment_params_disjoint.jsonl      ← 250 disjoint COP tasks
│
├── 📁 analyze_results/       ← Analysis and plotting package
│   ├── split_results.py              ← Split raw merged pkl into typed DataFrames
│   ├── compute_circuit_resources.py  ← Analytical gate-count table for all experiments
│   ├── compute_vcg_resources.py      ← Analytical gate-count table for trained VCGs
│   ├── generate_plots.py             ← Top-level orchestrator: combine splits, run analysis, copy to paper
│   ├── generate_results_markdown.py  ← GitHub-readable markdown tables of raw results
│   ├── main_analysis.py              ← Core analysis: plots, stats, paper summaries
│   ├── build_problem_table.py        ← Problem metadata table from raw results
│   ├── results_helper.py             ← ResultsCollector, CSV parsing, constraint remapping
│   ├── metrics.py                    ← P(feasible), P(optimal), AR_feas, summary stats
│   ├── plot_utils.py                 ← Shared matplotlib styling (rose-pine palette)
│   ├── plot_ar.py                    ← Approximation ratio plots
│   ├── plot_feasibility.py           ← P(feasible) and P(optimal) plots
│   ├── plot_resources.py             ← Circuit depth, shot budget, time breakdown
│   ├── plot_vcg_db.py                ← VCG database summary plots
│   ├── statistical_tests.py          ← Mann-Whitney U, Kruskal-Wallis significance tests
│   └── README.md
│
├── 📁 examples/
│   ├── example_vcg.py        ← VCG demo: train on a single constraint, plot counts
│   ├── example_pc_qaoa.py    ← PC-QAOA vs PenaltyQAOA on a three-constraint QUBO
│   ├── results/              ← Saved result pickles
│   └── figures/              ← Generated plots
│
├── 📁 tests/
│   ├── test_grover_decomp.py     ← Grover mixer decomposition tests
│   ├── test_flow_graph.py        ← Flow constraint graph tests
│   └── test_flow_state_prep.py   ← Flow state preparation tests
│
├── 📁 slurm/  (HPC)
│   ├── submit.sh           ← Full pipeline: submit VCG training + all experiments in dependency order
│   ├── vcg_train.sh        ← Single job: train all VCG gadgets (8 parallel workers)
│   ├── experiment_array.sh ← SLURM array: run one COP per task
│   ├── experiment_merge.sh ← Single job: merge pending results → pc_qaoa_vs_penalty.pkl
│   └── check_failed.sh     ← Utility: report failed tasks and print resubmit commands
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
│   └── qubos.csv             ← Random QUBOs, sizes 2–10 (10 per size)
│
├── 📁 gadgets/
│   └── vcg_db.pkl            ← Trained VCG gadget database
│
├── 📁 results/               ← Split result DataFrames and circuit resource tables
└── 📁 analysis_output/       ← Figures, stats, and summaries from analysis pipeline
```

## Quick Start

### Build and optimise a constraint gadget (Python)

```python
from core.vcg import VCG

gadget = VCG(
    constraints=["3*x_0 + 2*x_1 + x_2 <= 3"],
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

### Solve a constrained QUBO with PC-QAOA

```python
import numpy as np
from core import constraint_handler as ch
from core.pc_qaoa import PCQAOA

Q = np.array([[1, -2, 0], [-2, 3, -1], [0, -1, 2]], dtype=float)
constraints = ["x_0 + x_1 + x_2 == 1"]
parsed = ch.parse_constraints(constraints)

solver = PCQAOA(
    qubo=Q,
    all_constraints=parsed,
    structural_indices=[0],   # enforce via Dicke state prep
    penalty_indices=[],
    angle_strategy="ma-QAOA",
    n_layers=1,
    steps=50,
    num_restarts=10,
    gadget_db_path="gadgets/vcg_db.pkl",
)
opt_cost, opt_angles = solver.optimize_angles()
counts = solver.do_counts_circuit(shots=10_000)
```

### Run the toy examples

```bash
# VCG demo: train on 3*x_0 + 2*x_1 + x_2 <= 3, print AR / P(feasible), plot counts
python examples/example_vcg.py

# PC-QAOA vs PenaltyQAOA – three-constraint COP on 7 decision variables
python examples/example_pc_qaoa.py
```

## Running Experiments

### 1. Generate parameter files

```bash
# Extract unique VCG constraints from the experiment params
python run/generate_vcg_params.py

# Enumerate 250 overlapping + 250 disjoint COPs
python run/generate_experiment_params.py
```

### 2. Train the VCG gadget database

```bash
python run/create_vcg_database.py \
    --params run/params/vcg_params_experiments.jsonl \
    --db gadgets/vcg_db.pkl \
    --workers 8
```

### 3. Run PC-QAOA vs PenaltyQAOA experiments

```bash
# Sequential (local)
python run/run_pc_qaoa_vs_penalty.py \
    --params run/params/experiment_params_overlapping.jsonl \
    --db gadgets/vcg_db.pkl

# Merge SLURM results
python run/run_pc_qaoa_vs_penalty.py \
    --merge \
    --pending-dir results/pending_overlapping/ \
    --output results/overlapping/pc_qaoa_vs_penalty.pkl
```

### 4. Post-process and generate plots

```bash
# Split raw results into typed DataFrames (run once per split)
python analyze_results/split_results.py \
    --pc-qaoa results/overlapping/pc_qaoa_vs_penalty.pkl \
    --output-dir results/overlapping/

python analyze_results/split_results.py \
    --pc-qaoa results/disjoint/pc_qaoa_vs_penalty.pkl \
    --output-dir results/disjoint/

# Compute circuit resources
python analyze_results/compute_circuit_resources.py
python analyze_results/compute_vcg_resources.py

# Combine splits, generate all plots, and copy to paper/
python analyze_results/generate_plots.py
```

### SLURM (HPC)

```bash
# Submit VCG training + all 500 experiments in dependency order
bash slurm/submit.sh

# After merges complete, run post-processing manually
python analyze_results/split_results.py ...   # once per split
python analyze_results/compute_circuit_resources.py
python analyze_results/compute_vcg_resources.py
python analyze_results/generate_plots.py

# Check for failed tasks
bash slurm/check_failed.sh overlapping 250
bash slurm/check_failed.sh disjoint 250
```

## How the VCG Works

A **Variational Constraint Gadget (VCG)** is a small QAOA circuit whose
ground state is the uniform superposition over all bitstrings that satisfy a
given constraint.  Once trained, it acts as both the initial state and the
Grover mixer inside PC-QAOA, keeping the search within the feasible
subspace.  The Hamiltonian is defined directly on the decision-variable qubits
— no ancilla or flag qubit is used.

### Step 1 — Constraint Hamiltonian

VCG builds a diagonal Hamiltonian whose eigenvalues encode feasibility:

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

## VCG Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `constraints` | list[str] | — | Constraint strings, e.g. `["x_0 + x_1 == 1"]` |
| `ar_threshold` | float | `0.999` | Stop training when AR >= this value |
| `entropy_threshold` | float | `0.9999` | Stop when normalised entropy >= this value (once AR met) |
| `max_layers` | int | `8` | Maximum ma-QAOA layers in the sweep |
| `qaoa_restarts` | int | `5` | Random restarts for Stage 1 QAOA warm-start |
| `qaoa_steps` | int | `150` | Optimisation steps for Stage 1 QAOA warm-start |
| `ma_restarts` | int | `20` | Random restarts per ma-QAOA layer |
| `ma_steps` | int | `200` | Optimisation steps per ma-QAOA layer |
| `lr` | float | `0.05` | Adam learning rate |
| `samples` | int | `10_000` | Measurement shots for counts / P(feasible) |

## PC-QAOA Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `qubo` | np.ndarray | — | QUBO matrix (n_x × n_x) |
| `all_constraints` | list[ParsedConstraint] | — | All parsed constraints |
| `structural_indices` | list[int] | — | Indices enforced via gadget / Dicke prep |
| `penalty_indices` | list[int] | — | Indices enforced via penalty term |
| `angle_strategy` | str | `"ma-QAOA"` | `"QAOA"` or `"ma-QAOA"` |
| `n_layers` | int | `1` | QAOA circuit depth |
| `steps` | int | `50` | Optimisation steps per restart |
| `num_restarts` | int | `5` | Random restarts per layer |
| `gadget_db_path` | str | `None` | Path to trained VCG database pickle |
| `cqaoa_steps` | int | `30` | Steps for inline VCG training when gadget not in DB |
| `cqaoa_num_restarts` | int | `5` | Restarts for inline VCG training |

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

## Dependencies

```
pennylane >= 0.38
pennylane-lightning
numpy
pandas
matplotlib
seaborn
scipy
jax
```

Install with:

```bash
pip install pennylane pennylane-lightning numpy pandas matplotlib seaborn scipy jax
```

## Links

- [Analysis Package](analyze_results/README.md)
- [Run Scripts](run/README.md)
- [SLURM Pipeline](slurm/README.md)
