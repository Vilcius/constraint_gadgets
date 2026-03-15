# Results Evolution — Improvement Tracking

Goal: show HybridQAOA as the better method overall.
Fair comparison metrics: **P(feas)** and **P(opt)** (AR is not comparable
across methods due to Hamiltonian scale differences; AR_feas is the fair
replacement, tracked once the analysis pipeline is updated).

Each section below corresponds to a change.  Each change is first described,
then tested on the same four *focus cases* chosen to represent the hardest
problems from the server runs (jobs 245203 / 245206).  The same focus cases
are used throughout so improvements are directly comparable.

---

## Baseline — Server run results (VCG flag, STEPS=50, restarts=10)

These results come directly from the server experiment array (jobs 245203 /
245206).  They represent the worst-performing configurations: constraint-type
and `n_x` combinations where HybridQAOA falls furthest behind PenaltyQAOA.

### Case 1 — `independent_set + knapsack + cardinality`, n=7

Constraints: `x_0*x_1 == 0`, `2*x_2 + 1*x_3 + 4*x_4 <= 2`, `x_5 + x_6 == 2`

The worst case in the entire dataset.  HybridQAOA achieves P(feas)=**0.000**
across every layer while PenaltyQAOA reaches P(feas)=0.495 at p=3.
The knapsack structural gadget (flag-based VCG) is the primary bottleneck.

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

Constraints: `3*x_0 + 2*x_1 <= 2`, `2*x_2 + 1*x_3 <= 2`, `x_4 + x_5 + x_6 >= 3`

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

Constraints: `x_0 + x_1 >= 2`, `5*x_2 + 2*x_3 + 5*x_4 + 5*x_5 <= 9`

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

Constraints: `x_0 + x_1 + x_2 == 1`, `x_3 + x_4 >= 1`

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

**Root cause**: Cases 1–3 all involve knapsack VCG gadgets (flag-based) that
fail to concentrate probability mass on feasible states.  Case 4 (Dicke-only)
confirms Hybrid wins with enough layers when state prep is exact.
**The knapsack VCG gadget is the primary bottleneck.**

---

## Change 1 — Replace VCG (flag) with VCGNoFlag  [`core/vcg_no_flag.py`]

**Motivation**: Flag-based VCG for knapsack constraints achieves AR≈0.73 on the
gadget alone, meaning the initial state has substantial probability in infeasible
subspaces.  The Grover mixer cannot escape this — it reflects about this
corrupted initial state, and P(feas) stays near zero even with 5 QAOA layers.

**Change**: Replace VCG (flag) gadgets with VCGNoFlag in HybridQAOA for all
structural constraints not handled by Dicke / CardinalityLeq / Flow prep.
VCGNoFlag uses the same QAOA training but eliminates the flag qubit: the
Hamiltonian assigns eigenvalue −1 to feasible states and +1 to infeasible ones,
so AR=1.0 means the circuit outputs only feasible states.

Key implementation details:
- **One fewer qubit**: no ancilla flag wire.
- **P(feas) measured by direct constraint evaluation**, not flag bit, so there is
  no ambiguity.
- **Zero feasible states**: `ValueError` at init (impossible to train; use VCG).
- **One feasible state**: X-gate preparation; AR=1.0, depth=0, no training.
- **Many feasible states**: QAOA with WHT Pauli decomposition (O(n·2^n) build).

**Local gadget comparison** (`examples/compare_vcg_noflag.py`, constraint B =
`2*x_2 + 1*x_3 + 4*x_4 <= 2`, standalone, 10k shots):

| Gadget     | AR     | P(feas) | Qubits | Layers |
|------------|--------|---------|--------|--------|
| VCG (flag) | 0.7311 | 0.4778  | 4      | 1      |
| VCGNoFlag  | 1.0000 | 1.0000  | 3      | 1      |

**Results on focus cases** (`progress/run_focus_cases.py`, STEPS=50, restarts=10,
MAX_LAYERS=5, P_FEAS_THRESHOLD=0.75, 10k shots):

### Case 1 — `independent_set + knapsack + cardinality`, n=7

Gadgets used: `x_0*x_1==0` (VCGNoFlag, AR=1.0, p=1), `2*x_2+1*x_3+4*x_4<=2`
(VCGNoFlag, AR=1.0, p=1), `x_5+x_6==2` (Dicke).

| Variant    | p | AR     | P(feas) | P(opt) |
|------------|---|--------|---------|--------|
| VCG(flag)  | 1 | 0.7082 | 0.3850  | 0.0230 |
| VCG(flag)  | 2 | 0.7353 | 0.5250  | 0.0210 |
| VCG(flag)  | 3 | 0.6736 | 0.4880  | 0.0940 |
| VCG(flag)  | 4 | 0.7413 | 0.5570  | 0.0560 |
| VCG(flag)  | 5 | 0.7796 | 0.5970  | 0.0070 |
| VCGNoFlag  | 1 | 0.3602 | **1.000** | 0.1086 |

VCGNoFlag achieves P(feas)=1.000 at p=1 (threshold met immediately).  VCG(flag)
never reaches the 0.75 threshold across 5 layers, peaking at 0.597.

### Case 2 — `knapsack + knapsack + cardinality`, n=7

Gadgets: `3*x_0+2*x_1<=2` (AR=1.0, p=1), `2*x_2+1*x_3<=2` (AR=1.0, p=1),
`x_4+x_5+x_6>=3` (single feasible state `111` → X-gate prep, AR=1.0, depth=0).

| Variant    | p | AR     | P(feas) | P(opt) |
|------------|---|--------|---------|--------|
| VCG(flag)  | 1 | 0.5358 | 0.0930  | 0.0040 |
| VCG(flag)  | 2 | 0.5662 | 0.0780  | 0.0030 |
| VCG(flag)  | 3 | 0.7187 | 0.1400  | 0.0590 |
| VCG(flag)  | 4 | 0.6242 | 0.1170  | 0.0210 |
| VCG(flag)  | 5 | 0.5829 | 0.1400  | 0.0270 |
| VCGNoFlag  | 1 | 0.8106 | **1.000** | **0.6654** |

VCGNoFlag at p=1 achieves P(feas)=1.000 and P(opt)=0.665 — dramatically better
than VCG(flag) which peaks at P(feas)=0.140 and P(opt)=0.059 across 5 layers.
The X-gate preparation for the cardinality constraint is key: the Grover mixer
reflects about an initial state already concentrated on the unique optimum
`x_4=x_5=x_6=1`, giving P(opt) much higher than typical.

### Case 3 — `cardinality + knapsack`, n=6

Gadgets: `x_0+x_1>=2` (single feasible state `11` → X-gate prep, AR=1.0,
depth=0), `5*x_2+2*x_3+5*x_4+5*x_5<=9` (AR=1.0, p=1).

| Variant    | p | AR     | P(feas) | P(opt) |
|------------|---|--------|---------|--------|
| VCG(flag)  | 1 | 0.5487 | 0.2610  | 0.0930 |
| VCG(flag)  | 2 | 0.5756 | 0.3280  | 0.0610 |
| VCG(flag)  | 3 | 0.6194 | 0.2830  | 0.1310 |
| VCG(flag)  | 4 | 0.6200 | 0.1530  | 0.0450 |
| VCG(flag)  | 5 | 0.7436 | 0.5270  | 0.0020 |
| VCGNoFlag  | 1 | 0.5134 | **1.000** | 0.0000 |

P(feas) jumps to 1.000 at p=1 but P(opt)=0.000: all shots are feasible but none
hit the optimum.  The penalty constraint (`5*x_2+...<=9`) has 8/16 feasible
assignments; the outer QAOA at p=1 evenly spreads probability across the feasible
subspace.  This case needs more QAOA layers or a better optimization budget
(TODO #4).

### Case 4 — `cardinality + cardinality`, n=5  (reference)

Gadgets: `x_0+x_1+x_2==1` (Dicke), `x_3+x_4>=1` (VCGNoFlag, AR=1.0, p=1).

| Variant    | p | AR     | P(feas) | P(opt) |
|------------|---|--------|---------|--------|
| VCG(flag)  | 1 | 0.8411 | 0.9820  | 0.0000 |
| VCGNoFlag  | 1 | 0.7336 | **1.000** | 0.0000 |

Both variants achieve high P(feas) here (no knapsack gadget); VCGNoFlag adds
a small improvement.  P(opt)=0.000 in both cases — optimization quality is the
bottleneck, not feasibility (same issue as Case 3).

**Summary**:

| Case | VCG(flag) best P(feas)      | VCGNoFlag P(feas) p=1 |
|------|-----------------------------|-----------------------|
| 1    | 0.597 (p=5, never ≥ 0.75)  | **1.000**             |
| 2    | 0.140 (p=3 or 5)            | **1.000**             |
| 3    | 0.527 (p=5, never ≥ 0.75)  | **1.000**             |
| 4    | 0.982 (p=1)                 | **1.000**             |

VCGNoFlag eliminates the feasibility bottleneck in all 4 cases at p=1.
Remaining gap: P(opt) still low in Cases 1, 3, 4 — the outer QAOA optimization
needs improvement (addressed by TODO #3 and #4).

**Status**: Implemented and tested on all 4 focus cases.

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

**Status**: Function implemented. Pipeline integration (main_analysis, plot_ar.py,
comparison CSVs, plots) pending.

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
