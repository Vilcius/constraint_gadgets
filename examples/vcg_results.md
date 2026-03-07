# VCG Training: QAOA vs ma-QAOA — Current State and Recommendations

> **Status:** Initial sweep complete on 5-variable linear and quadratic
> knapsack constraints.  Results, figures, and the full sweep script live in
> `examples/test_vcg_layers.py` and `examples/results/vcg_layer_sweep.pkl`.

---

## 1. What is a VCG?

A **Variational Constraint Gadget (VCG)** is a small QAOA circuit trained so
that its ground state is the uniform superposition over all bitstrings
satisfying a given constraint.  Once trained it is used inside HybridQAOA as:

- the **initial state** (structured state-preparation replacing |+⟩^n), and
- the **Grover mixer** (reflecting about the feasible subspace).

This keeps the outer QAOA search within the feasible region from the start,
rather than penalising infeasibility in the cost Hamiltonian.

---

## 2. VCG Construction Process

### Step 1 — Parse the constraint

The constraint string (e.g. `5*x_0 + 10*x_1 + 1*x_2 + 9*x_3 + 6*x_4 <= 19`)
is parsed into coefficient maps plus an operator and RHS.  Variable indices
determine which wires carry the decision variables.  One additional **flag
qubit** per constraint marks whether an assignment satisfies the constraint.

### Step 2 — Build the constraint Hamiltonian

A truth table is built over all `2^(n_x + n_c)` states:

- For each assignment of the `n_x` decision variables, evaluate the constraint
  and set the flag qubit to 0 (satisfied) or 1 (violated).
- Label every *valid* state (variables + matching flag) with eigenvalue **−1**;
  label every *invalid* state (impossible flag assignment) with **+1**.
- Compute the Pauli decomposition via a **Walsh-Hadamard transform (WHT)**:

```
H_constraint = Σ_k  w_k · P_k    (Z-only Pauli terms)

c_S = (1/2^n) · Σ_x  outcomes[x] · (−1)^{popcount(x & S)}
```

Because the Hamiltonian is diagonal, all Pauli terms are products of Z
operators and therefore **mutually commute**.  The WHT computes their
coefficients in O(n · 2^n) time and O(2^n) memory — far more efficient than
constructing the full `2^n × 2^n` matrix and calling `qml.pauli_decompose`
(O(4^n) in both time and memory, which runs out of RAM for n ≳ 13).

The number of non-trivial terms (`num_gamma`) drives the cost-layer gate
count and determines how many independent angles ma-QAOA needs.

> For the 5-variable knapsack tested here: 6 qubits total (5 vars + 1 flag),
> 64 states, 32 good / 32 bad, **32 Pauli terms**.

### Step 3 — Build the QAOA circuit

```
|+⟩^n → [Cost(γ) · Mixer(β)]^p → measure
```

| Component | Details |
|---|---|
| Initialisation | Hadamard on every qubit |
| Cost layer | `MultiRZ(w_k · γ_k, wires)` for each non-identity Pauli term k |
| Mixer layer | `RX(β_i, wire_i)` for each qubit i (X-mixer) |
| Depth | p = `n_layers` repetitions |

For `decompose=False` (QAOA only), the cost layer is instead a single
`DiagonalQubitUnitary(exp(−iγ · outcomes))` — mathematically equivalent since
all Z-terms commute, but incompatible with ma-QAOA which needs per-term angles.

### Step 4 — Angle strategies

| Strategy | Params per layer | Free angles |
|---|---|---|
| **QAOA** | 2 | One shared γ (all Pauli terms), one shared β (all qubits) |
| **ma-QAOA** | `num_gamma + num_beta` | Independent γ per Pauli term, β per qubit |

QAOA is a **special case** of ma-QAOA where all γ values are forced equal and
all β values are forced equal.  ma-QAOA's optimal cost is therefore always
≤ QAOA's optimal cost (ma-QAOA can only do better or the same).

For the 5-variable knapsack: QAOA has 2 params/layer, ma-QAOA has
32 + 6 = **38 params/layer**.

### Step 5 — Optimisation

**Step 5a — single QAOA run at p=1 (fast warm-up, ~8 s):**

A standard QAOA circuit with 2 free parameters (one shared γ, one shared β)
is optimised first.  Its optimal angles are then broadcast to ma-QAOA format
(γ → all `num_gamma` entries, β → all `num_beta` entries) to seed the first
restart at ma-QAOA p=1.

**Step 5b — ma-QAOA layer sweep:**

Starting from the QAOA warm-start, ma-QAOA is optimised at p=1 with 20
restarts × 200 steps.  If AR < threshold, a new layer is added with joint
re-optimisation of all p×k parameters, warm-started from the previous depth:

| Phase | Method | Restarts | Steps |
|---|---|---|---|
| Warm-up | QAOA p=1 | 5 | 150 |
| Sweep | ma-QAOA p=1,2,... | 20 | 200 |

This guarantees ma-QAOA starts from a point at least as good as QAOA and
only improves.

---

## 3. Results

### Summary table

| Constraint | QAOA p=1 AR | ma-QAOA p* | ma-QAOA AR | ma-QAOA time |
|---|---|---|---|---|
| knapsack      | 0.900 | **1** | **1.0000** | ~506 s |
| quad_knapsack | 0.917 | **1** | **1.0000** | ~500 s |

AR threshold: 0.95.  ma-QAOA hits the exact ground state at p=1 in both
cases, confirming that the QAOA warm-start is sufficient to find the global
optimum.

### AR vs circuit depth

![AR vs layers](figures/vcg_layer_sweep_ar.png)

QAOA reaches the threshold by p=2–3 and saturates — adding more layers
provides diminishing returns.  ma-QAOA hits AR=1.0 at p=1 and stops.  The
QAOA ceiling is real: the shared-angle constraint limits expressibility
regardless of depth.

### Optimisation time vs circuit depth

![Time vs layers](figures/vcg_layer_sweep_time.png)

QAOA is fast: ~8–10 s at p=1, accumulating ~25–50 s by the threshold layer.
ma-QAOA takes ~500 s at p=1 due to the larger parameter space and higher
restart count.  With joint optimisation, ma-QAOA time grows with depth since
all layers are optimised together.

### Measurement distributions (best layer per run)

![Distributions](figures/vcg_layer_sweep_distributions.png)

For ma-QAOA the good (foam) states carry essentially all probability mass,
confirming AR=1.0.  For QAOA the distribution is wider but clearly biased
toward good states.

---

## 4. Key Findings

1. **ma-QAOA strictly dominates QAOA on expressibility** — AR=1.0 at p=1 vs
   AR≈0.985 at p=2 for QAOA.  This is expected theoretically (QAOA ⊂ ma-QAOA)
   but required proper optimisation to observe empirically.

2. **QAOA has a structural ceiling** — even with joint optimisation and
   unlimited depth, the shared-angle constraint caps AR below 1.0 for these
   Hamiltonians.  The ceiling is ~0.985, not 1.0, because the single γ cannot
   independently weight all 32 Pauli terms.

3. **Warm-starting ma-QAOA from QAOA is critical** — without it, 20 restarts
   in a 38-dimensional space is insufficient to reliably find the ground
   state.  With it, ma-QAOA converges reliably at p=1.

5. **Training cost is a one-time expense** — the ~500–510 s ma-QAOA training is
   cached in `GadgetDatabase`.  HybridQAOA looks up the stored gadget and
   reuses it at no additional cost.

---

## 5. Recommended Approach

Use `run/add_to_vcg_database.py` for production VCG creation.  It runs both
sweeps automatically and stores the best ma-QAOA result:

```bash
python run/add_to_vcg_database.py \
    --constraints "6*x_0 + 2*x_1 + 2*x_2 <= 3" \
    --db gadgets/gadget_db.pkl
```

Internally, the sweep runs QAOA (warm-starts ma-QAOA at p=1), then ma-QAOA
with joint optimisation of all layers:

```python
# QAOA sweep – joint re-opt of all layers, warm-started from previous depth
opt_cost, qaoa_p1_angles = gadget_qaoa.optimize_angles(
    gadget_qaoa.do_evolution_circuit,
    prev_layer_angles=prev_best,   # None at p=1
)

# ma-QAOA p=1 – seeded from QAOA p=1 solution
opt_cost, _ = gadget_ma.optimize_angles(
    gadget_ma.do_evolution_circuit,
    starting_angles_from_qaoa=qaoa_p1_angles,
)

# ma-QAOA p>1 – joint re-opt, warm-started from previous depth
opt_cost, _ = gadget_ma.optimize_angles(
    gadget_ma.do_evolution_circuit,
    prev_layer_angles=prev_best_ma,
)
```

**Decision guide:**

| Situation | Recommended strategy |
|---|---|
| Need AR ≥ 0.95, time budget < 1 min | QAOA p=2, joint re-opt |
| Need AR = 1.0 or maximum quality | ma-QAOA p=1, QAOA warm-start |
| Constraint has few Pauli terms (< 10) | Either; both will converge quickly |
| Constraint has many Pauli terms (> 20) | ma-QAOA with scaled restarts |
| Gadget will be reused in HybridQAOA | Invest in ma-QAOA; pay once, reuse many times |

---

## 6. Open Questions / Future Work

- **Scaling to larger n** — all tests here are n=5 variables.  The
  Hamiltonian construction bottleneck (previously O(4^n) via
  `qml.pauli_decompose` on a full matrix) is now resolved: the implementation
  uses a Walsh-Hadamard transform (WHT) that runs in O(n · 2^n) time and
  O(2^n) memory, making construction negligible even at n=15+.  The remaining
  bottleneck is truth table enumeration (2^n states) and the optimisation
  itself, which grow with n_x.

- **Optimiser alternatives** — Adam with random restarts works but is not
  ideal for a 38-dimensional landscape.  L-BFGS-B (gradient + Hessian
  approximation) or SPSA (gradient-free, noise-robust) could converge faster.
  Bayesian optimisation over the restart space is another avenue.

- **Coefficient-magnitude pruning** — the 32 Pauli terms have varying weights.
  Dropping near-zero terms reduces gate count and parameter count, potentially
  making the landscape easier to optimise.  Worth benchmarking for larger n
  where the full decomposition becomes expensive.

- **Warm-start depth transfer** — for p>1, the warm-start from the previous
  depth is the primary mechanism for seeding new layers.  Experimenting with
  QAOA-seeded initialisations at each new depth may improve convergence for
  hard constraints.
