# Results Evolution — Improvement Tracking

Goal: show HybridQAOA as the better method overall.
Fair comparison metrics: **P(feas)** and **P(opt)** (AR is not comparable
across methods due to Hamiltonian scale differences; AR_feas is the fair
replacement, tracked once the analysis pipeline is updated).

Each section below corresponds to a TODO item.  Results are shown on a fixed
set of *focus cases* chosen to represent the hardest problems from the server
runs (job 245203 / 245206).  The same focus cases will be used for every
entry so improvements are directly comparable.

---

## Focus cases (from server runs, flag-based VCG, budget: STEPS=50, restarts=10)

These are the constraint-type / n_x / p combinations where HybridQAOA
performs worst relative to PenaltyQAOA.

### Case 1 — `independent_set + knapsack + cardinality`, n=7
The worst case in the entire dataset.  HybridQAOA achieves P(feas)=**0.000**
across every layer while PenaltyQAOA reaches P(feas)=0.495 at p=3.
The knapsack structural gadget (flag-based VCG) is likely the bottleneck.

| Method      | p | AR    | P(feas) | P(opt) |
|-------------|---|-------|---------|--------|
| HybridQAOA  | 1 | 0.413 | 0.000   | 0.000  |
| HybridQAOA  | 2 | 0.541 | 0.000   | 0.000  |
| HybridQAOA  | 3 | 0.600 | 0.000   | 0.000  |
| HybridQAOA  | 4 | 0.598 | 0.000   | 0.000  |
| HybridQAOA  | 5 | 0.570 | 0.000   | 0.000  |
| PenaltyQAOA | 1 | 0.918 | 0.153   | 0.009  |
| PenaltyQAOA | 2 | 0.932 | 0.428   | 0.027  |
| PenaltyQAOA | 3 | 0.969 | 0.495   | 0.043  |
| PenaltyQAOA | 4 | 0.919 | 0.153   | 0.003  |
| PenaltyQAOA | 5 | 0.851 | 0.071   | 0.012  |

### Case 2 — `knapsack + knapsack + cardinality`, n=7
Two knapsack gadgets (both flag-based VCG).  HybridQAOA peaks at
P(feas)=0.193.  PenaltyQAOA reaches P(feas)=0.781 at p=2.

| Method      | p | AR    | P(feas) | P(opt) |
|-------------|---|-------|---------|--------|
| HybridQAOA  | 1 | 0.567 | 0.048   | 0.016  |
| HybridQAOA  | 2 | 0.543 | 0.056   | 0.014  |
| HybridQAOA  | 3 | 0.655 | 0.177   | 0.032  |
| HybridQAOA  | 4 | 0.716 | 0.193   | 0.037  |
| HybridQAOA  | 5 | 0.739 | 0.122   | 0.024  |
| PenaltyQAOA | 1 | 0.930 | 0.225   | 0.009  |
| PenaltyQAOA | 2 | 0.977 | 0.781   | 0.003  |

### Case 3 — `cardinality + knapsack`, n=6
One knapsack VCG gadget.  P(feas)=0.000 at n=6 for every HybridQAOA layer.
PenaltyQAOA climbs to P(feas)=0.886 at p=3.

| Method      | p | AR    | P(feas) | P(opt) |
|-------------|---|-------|---------|--------|
| HybridQAOA  | 1 | 0.347 | 0.000   | 0.000  |
| HybridQAOA  | 2 | 0.421 | 0.000   | 0.000  |
| HybridQAOA  | 3 | 0.481 | 0.000   | 0.000  |
| HybridQAOA  | 4 | 0.507 | 0.000   | 0.000  |
| HybridQAOA  | 5 | 0.500 | 0.000   | 0.000  |
| PenaltyQAOA | 1 | 0.939 | 0.386   | 0.003  |
| PenaltyQAOA | 2 | 0.984 | 0.692   | 0.690  |
| PenaltyQAOA | 3 | 0.992 | 0.886   | 0.871  |

### Case 4 — `cardinality + cardinality`, n=5  (reference: Hybrid eventually wins)
Both structural constraints are Dicke-compatible (no VCG gadget).
HybridQAOA improves with depth and reaches P(feas)=0.732 at p=5, though
PenaltyQAOA is still stronger at p=2 (P(feas)=0.943).  Shows the pattern
where Hybrid wins given enough layers.

| Method      | p | AR    | P(feas) | P(opt) |
|-------------|---|-------|---------|--------|
| HybridQAOA  | 1 | 0.572 | 0.241   | 0.106  |
| HybridQAOA  | 2 | 0.589 | 0.080   | 0.051  |
| HybridQAOA  | 3 | 0.810 | 0.584   | 0.459  |
| HybridQAOA  | 4 | 0.844 | 0.438   | 0.399  |
| HybridQAOA  | 5 | 0.889 | 0.732   | 0.685  |
| PenaltyQAOA | 1 | 0.923 | 0.532   | 0.261  |
| PenaltyQAOA | 2 | 0.989 | 0.943   | 0.929  |

**Pattern**: Cases 1–3 all involve knapsack VCG gadgets (flag-based) that fail
to produce feasible initial states.  Case 4 (Dicke-only) shows Hybrid can win
with enough layers.  **The knapsack VCG gadget is the primary bottleneck.**

---

## Change 1 — Replace VCG (flag) with VCGNoFlag  [`core/vcg_no_flag.py`]

**Motivation**: Flag-based VCG for constraint B achieves AR=0.73, P(feas)=0.48
on the gadget itself.  VCGNoFlag achieves AR=1.0, P(feas)=1.0 at the same
depth (p=1) with one fewer qubit.

**Local test** (`examples/compare_vcg_noflag.py`, 3-constraint problem,
n=7, p=1, STEPS=50, restarts=10):

| Gadget (Constraint B standalone) | AR     | P(feas) | Qubits | Layers |
|----------------------------------|--------|---------|--------|--------|
| VCG (flag)                       | 0.7311 | 0.4778  | 4      | 1      |
| VCGNoFlag                        | 1.0000 | 1.0000  | 3      | 1      |

| Full solver (HybridQAOA, p=1)    | AR     | P(feas) | P(opt) | Qubits | Time(s) |
|----------------------------------|--------|---------|--------|--------|---------|
| VCG (flag)                       | 0.8958 | 0.1938  | 0.0007 | 9      | 23.8    |
| VCGNoFlag                        | 0.9553 | 1.0000  | 0.0180 | 8      | 34.4    |

**Impact on focus cases (projected)**: Cases 1–3 all use knapsack VCG gadgets.
VCGNoFlag should raise P(feas) from 0.000 toward 1.0 for the gadget component.
The full-problem P(feas) is also gated by: PenaltyQAOA constraint C and the
outer QAOA quality — P(opt) improvement will require TODO items 3 and 4.

**Edge cases handled**:
- 0 feasible states → `ValueError` at init (was: silent AR=0/0 corruption)
- 1 feasible state → X-gate preparation, AR=1.0, depth=0, no training needed
  (was: QAOA would fail to concentrate on a single basis state)

**Status**: Implemented. Awaiting server re-run to get full focus-case numbers.

---

## Change 2 — Feasibility-conditioned AR metric  [`analyze_results/metrics.py`]  *(TODO #1)*

**Motivation**: Raw AR is not comparable — PenaltyQAOA inflates C_max by
~3000–4000× via penalty terms, making its AR appear artificially high
(~0.93) vs HybridQAOA (~0.57) even when P(feas) and P(opt) favor Hybrid.

**Formula** (coauthor's):

    AR_feas = (f_max_F - E[f(x) : x ∈ F]) / (f_max_F - f*)

where F is the feasible set, f* is the true optimum, f_max_F is the worst
feasible QUBO value, and expectation is over feasible shots only.

**Implemented**: `ar_feasibility_conditioned()` and `compute_feasible_range()`
in `analyze_results/metrics.py`.

**Caveats**:
- Undefined (NaN) when P(feas)=0 — report alongside P(feas) always.
- Ignores P(feas) entirely: a method with 1 lucky feasible shot gets AR_feas=1.0.
- f_max_F requires brute-force enumeration (O(2^n_x)); fine for n≤12.

**Status**: Function implemented. Pipeline integration (main_analysis, plots,
CSVs) pending.

---

## Change 3 — P(feas) vs layer / P(opt) vs layer figures  *(TODO #2)*

*Pending.*

---

## Change 4 — Principled penalty weight assignment  *(TODO #3)*

*Pending.*

---

## Change 5 — Increase optimization budget  *(TODO #4)*

*Pending.*

---

## Change 6 — Hamiltonian normalization for PenaltyQAOA  *(TODO #6)*

**Motivation**: PenaltyQAOA's Hamiltonian range is ~3000–4000× larger than
HybridQAOA's, making the optimizer face a harder landscape with the same
step size and budget.  Normalizing to [-1, 1] before optimization makes the
comparison fair.

*Pending.*
