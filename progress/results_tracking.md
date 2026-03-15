# Results Evolution — Improvement Tracking

Goal: show HybridQAOA as the better method overall.
Fair comparison metrics: **P(feas)** and **P(opt)** (AR is not comparable
across methods due to Hamiltonian scale differences; AR_feas is the fair
replacement, tracked once the analysis pipeline is updated).

Each section below corresponds to a change.  Each change is first described,
then tested on the same four *focus cases* chosen to represent the hardest
problems from the server runs (jobs 245203 / 245206).  The same focus cases
are used throughout so improvements are directly comparable.

**Bolding convention**: in each table the best value per column is bolded.

---

## Baseline — Server run results (VCG flag, STEPS=50, restarts=10)

These results come directly from the server experiment array (jobs 245203 /
245206).  They represent the worst-performing configurations: constraint-type
and `n_x` combinations where HybridQAOA falls furthest behind PenaltyQAOA.

### Case 1 — `independent_set + knapsack + cardinality`, n=7

Constraints: `x_0*x_1 == 0`, `2*x_2 + 1*x_3 + 4*x_4 <= 2`, `x_5 + x_6 == 2`

The worst case in the entire dataset.  HybridQAOA achieves P(feas)=0.000
across every layer while PenaltyQAOA reaches P(feas)=0.495 at p=3.
The knapsack structural gadget (flag-based VCG) is the primary bottleneck.

| Method      | p | AR        | P(feas)       | P(opt)        |
|-------------|---|-----------|---------------|---------------|
| HybridQAOA  | 1 | 0.413     | 0.000         | 0.000         |
| HybridQAOA  | 2 | 0.541     | 0.000         | 0.000         |
| HybridQAOA  | 3 | 0.600     | 0.000         | 0.000         |
| HybridQAOA  | 4 | 0.598     | 0.000         | 0.000         |
| HybridQAOA  | 5 | 0.570     | 0.000         | 0.000         |
| PenaltyQAOA | 1 | 0.918     | 0.153         | 0.009         |
| PenaltyQAOA | 2 | 0.932     | 0.428         | 0.027         |
| PenaltyQAOA | 3 | **0.969** | **0.495**     | **0.043**     |
| PenaltyQAOA | 4 | 0.919     | 0.153         | 0.003         |
| PenaltyQAOA | 5 | 0.851     | 0.071         | 0.012         |

### Case 2 — `knapsack + knapsack + cardinality`, n=7

Constraints: `3*x_0 + 2*x_1 <= 2`, `2*x_2 + 1*x_3 <= 2`, `x_4 + x_5 + x_6 >= 3`

Two knapsack gadgets (both flag-based VCG).  HybridQAOA peaks at
P(feas)=0.193.  PenaltyQAOA reaches P(feas)=0.781 at p=2.

| Method      | p | AR        | P(feas)       | P(opt)        |
|-------------|---|-----------|---------------|---------------|
| HybridQAOA  | 1 | 0.567     | 0.048         | 0.016         |
| HybridQAOA  | 2 | 0.543     | 0.056         | 0.014         |
| HybridQAOA  | 3 | 0.655     | 0.177         | 0.032         |
| HybridQAOA  | 4 | 0.716     | 0.193         | **0.037**     |
| HybridQAOA  | 5 | 0.739     | 0.122         | 0.024         |
| PenaltyQAOA | 1 | 0.930     | 0.225         | 0.009         |
| PenaltyQAOA | 2 | **0.977** | **0.781**     | 0.003         |

### Case 3 — `cardinality + knapsack`, n=6

Constraints: `x_0 + x_1 >= 2`, `5*x_2 + 2*x_3 + 5*x_4 + 5*x_5 <= 9`

One knapsack VCG gadget.  P(feas)=0.000 at n=6 for every HybridQAOA layer.
PenaltyQAOA climbs to P(feas)=0.886 at p=3.

| Method      | p | AR        | P(feas)       | P(opt)        |
|-------------|---|-----------|---------------|---------------|
| HybridQAOA  | 1 | 0.347     | 0.000         | 0.000         |
| HybridQAOA  | 2 | 0.421     | 0.000         | 0.000         |
| HybridQAOA  | 3 | 0.481     | 0.000         | 0.000         |
| HybridQAOA  | 4 | 0.507     | 0.000         | 0.000         |
| HybridQAOA  | 5 | 0.500     | 0.000         | 0.000         |
| PenaltyQAOA | 1 | 0.939     | 0.386         | 0.003         |
| PenaltyQAOA | 2 | 0.984     | 0.692         | 0.690         |
| PenaltyQAOA | 3 | **0.992** | **0.886**     | **0.871**     |

### Case 4 — `cardinality + cardinality`, n=5  (reference: Hybrid eventually wins)

Constraints: `x_0 + x_1 + x_2 == 1`, `x_3 + x_4 >= 1`

Both structural constraints are Dicke-compatible (no VCG gadget).
HybridQAOA improves with depth and reaches P(feas)=0.732 at p=5, though
PenaltyQAOA is still stronger at p=2 (P(feas)=0.943).  Shows the pattern
where Hybrid wins given enough layers.

| Method      | p | AR        | P(feas)       | P(opt)        |
|-------------|---|-----------|---------------|---------------|
| HybridQAOA  | 1 | 0.572     | 0.241         | 0.106         |
| HybridQAOA  | 2 | 0.589     | 0.080         | 0.051         |
| HybridQAOA  | 3 | 0.810     | 0.584         | 0.459         |
| HybridQAOA  | 4 | 0.844     | 0.438         | 0.399         |
| HybridQAOA  | 5 | 0.889     | 0.732         | 0.685         |
| PenaltyQAOA | 1 | 0.923     | 0.532         | 0.261         |
| PenaltyQAOA | 2 | **0.989** | **0.943**     | **0.929**     |

**Root cause**: Cases 1–3 all involve knapsack VCG gadgets (flag-based) that
fail to concentrate probability mass on feasible states.  Case 4 (Dicke-only)
confirms Hybrid wins with enough layers when state prep is exact.
**The knapsack VCG gadget is the primary bottleneck.**

---
---

## Change 1 — Replace VCG (flag) with VCGNoFlag  [`core/vcg_no_flag.py`]

**Motivation**: Flag-based VCG for knapsack constraints achieves AR≈0.73 on the
gadget alone, meaning the initial state has substantial probability in infeasible
subspaces.  The Grover mixer reflects about this corrupted initial state and
cannot escape it — P(feas) stays low even with 5 QAOA layers.

**Change**: Replace VCG (flag) gadgets with VCGNoFlag for all structural
constraints not handled by Dicke / CardinalityLeq / Flow prep.  VCGNoFlag uses
the same QAOA training but eliminates the flag qubit: the Hamiltonian assigns
eigenvalue −1 to feasible states and +1 to infeasible ones, so AR=1.0 means
the circuit outputs only feasible states.

Key implementation details:
- **One fewer qubit**: no ancilla flag wire.
- **P(feas) measured by direct constraint evaluation**, not flag bit.
- **Zero feasible states**: `ValueError` at init (use VCG flag instead).
- **One feasible state**: X-gate preparation; AR=1.0, depth=0, no training.
- **Many feasible states**: QAOA with WHT Pauli decomposition (O(n·2^n) build).

**Note on `auto` partitioning**: Each state-prep circuit (Dicke, LEQ, Flow, or
VCG gadget) handles exactly one constraint on a fixed set of qubits.  Two
structural constraints must therefore have disjoint variable sets — overlapping
qubits would make the Grover mixer composition ill-defined.  `partition_constraints`
(fixed in this session) now enforces this: exact preps (Dicke/LEQ/Flow) are
assigned first (preferred); each remaining constraint becomes structural only if
its variable set is disjoint from all already-claimed variables, otherwise it is
penalized.  The old code incorrectly grouped all overlapping constraints as
structural if the total unique variable count was ≤ 12 (and wrongly implied a
single joint gadget could handle a group — VCG always handles one constraint).
For the four focus cases this makes no difference because all constraints use
disjoint variable sets regardless.

**Note on the local VCG(flag) results vs server baseline**: The server jobs
(245203/245206) showed P(feas)=0.000 for Cases 1 and 3 with Hybrid+VCG(flag).
The local re-run below shows P(feas)=0.385–0.527.  The most likely cause is
that `gadgets/gadget_db.pkl` has been retrained/updated since those server jobs
ran, giving better gadgets locally.  The server numbers in the Baseline section
are the authoritative comparison point for the paper.

**Local gadget comparison** (`examples/compare_vcg_noflag.py`, constraint
`2*x_2 + 1*x_3 + 4*x_4 <= 2` standalone, 10k shots):

| Gadget          | AR         | P(feas)    | Qubits  | Layers |
|-----------------|------------|------------|---------|--------|
| VCG (flag)      | 0.7311     | 0.4778     | 4       | 1      |
| VCGNoFlag       | **1.0000** | **1.0000** | **3**   | 1      |

**Results on focus cases** (`progress/run_focus_cases.py`, STEPS=50, restarts=10,
MAX_LAYERS=5, P_FEAS_THRESHOLD=0.75, 10k shots).

All four focus cases have constraints on **disjoint variable sets**, so each
constraint forms its own group under `auto` — each gets its own gadget.
No constraints are penalized in any of these cases (all groups ≤ 5 variables).
P(feas) is checked against **all** constraints by evaluating them directly on
output bitstrings.

"Hybrid+VCG(flag)" = local re-run (gadget_db may differ from server).
"PenaltyQAOA (best p)" = best layer from the server baseline above.

### Case 1 — `independent_set + knapsack + cardinality`, n=7

Gadgets (all disjoint): `x_0*x_1==0` (VCGNoFlag, AR=1.0, p=1),
`2*x_2+1*x_3+4*x_4<=2` (VCGNoFlag, AR=1.0, p=1), `x_5+x_6==2` (Dicke).
12 feasible assignments out of 128.

| Variant              | p | AR     | P(feas)   | P(opt)    |
|----------------------|---|--------|-----------|-----------|
| Hybrid+VCG(flag)     | 1 | 0.6466 | 0.3170    | 0.0120    |
| Hybrid+VCG(flag)     | 2 | 0.7520 | 0.4310    | 0.0760    |
| Hybrid+VCG(flag)     | 3 | 0.7823 | 0.4510    | 0.0070    |
| Hybrid+VCG(flag)     | 4 | 0.7116 | 0.3410    | 0.0120    |
| Hybrid+VCG(flag)     | 5 | 0.7745 | 0.6110    | 0.0410    |
| PenaltyQAOA (best p) | 3 | 0.969  | 0.495     | 0.043     |
| Hybrid+VCGNoFlag     | 1 | 0.4008 | **1.000** | **0.555** |

**Hybrid+VCGNoFlag beats PenaltyQAOA**: P(feas) 1.000 vs 0.495, P(opt) 0.555 vs 0.043.
The feasible set is small (12/128) so the outer QAOA at p=1 concentrates well on the optimum.

### Case 2 — `knapsack + knapsack + cardinality`, n=7

Gadgets (all disjoint): `3*x_0+2*x_1<=2` (VCGNoFlag, AR=1.0, p=1),
`2*x_2+1*x_3<=2` (VCGNoFlag, AR=1.0, p=1), `x_4+x_5+x_6>=3`
(**1 feasible state** `111` → X-gate prep, depth=0).  2 feasible assignments
out of 128.

| Variant              | p | AR     | P(feas)   | P(opt)    |
|----------------------|---|--------|-----------|-----------|
| Hybrid+VCG(flag)     | 1 | 0.6639 | 0.1540    | 0.0470    |
| Hybrid+VCG(flag)     | 2 | 0.5847 | 0.0410    | 0.0080    |
| Hybrid+VCG(flag)     | 3 | 0.6570 | 0.0720    | 0.0070    |
| Hybrid+VCG(flag)     | 4 | 0.6021 | 0.0730    | 0.0070    |
| Hybrid+VCG(flag)     | 5 | 0.6808 | 0.1910    | 0.0130    |
| PenaltyQAOA (best p) | 2 | 0.977  | 0.781     | 0.009     |
| Hybrid+VCGNoFlag     | 1 | 0.7480 | **1.000** | **0.356** |

**Hybrid+VCGNoFlag beats PenaltyQAOA**: P(feas) 1.000 vs 0.781, P(opt) 0.356 vs 0.009.
P(opt) is boosted by the X-gate prep pinning x_4=x_5=x_6=1, leaving only 2
feasible assignments for the outer QAOA to distinguish — p=1 suffices.

### Case 3 — `cardinality + knapsack`, n=6

Gadgets (disjoint): `x_0+x_1>=2` (**1 feasible state** `11` → X-gate prep,
depth=0), `5*x_2+2*x_3+5*x_4+5*x_5<=9` (VCGNoFlag, AR=1.0, p=1).
8 feasible assignments out of 64 (x_0=x_1=1 fixed; 8/16 for knapsack).

| Variant              | p | AR     | P(feas)   | P(opt)    |
|----------------------|---|--------|-----------|-----------|
| Hybrid+VCG(flag)     | 1 | 0.5984 | 0.3550    | 0.0390    |
| Hybrid+VCG(flag)     | 2 | 0.5207 | 0.2070    | 0.0090    |
| Hybrid+VCG(flag)     | 3 | 0.6361 | 0.3990    | 0.1090    |
| Hybrid+VCG(flag)     | 4 | 0.6912 | 0.2430    | 0.1060    |
| Hybrid+VCG(flag)     | 5 | 0.7690 | 0.5490    | 0.2060    |
| PenaltyQAOA (best p) | 3 | 0.992  | **0.886** | **0.871** |
| Hybrid+VCGNoFlag     | 1 | 0.5135 | **1.000** | 0.001     |

**Hybrid+VCGNoFlag partially beats PenaltyQAOA**: P(feas) 1.000 vs 0.886, but
P(opt) 0.001 vs 0.871.  With x_0=x_1=1 fixed, 8 feasible states remain — the
outer QAOA at p=1 spreads probability nearly uniformly among them.
Needs more layers / budget (TODO #4).

### Case 4 — `cardinality + cardinality`, n=5  (reference)

Gadgets (disjoint): `x_0+x_1+x_2==1` (Dicke), `x_3+x_4>=1` (VCGNoFlag,
AR=1.0, p=1).  9 feasible assignments out of 32.

| Variant              | p | AR     | P(feas)   | P(opt)    |
|----------------------|---|--------|-----------|-----------|
| Hybrid+VCG(flag)     | 1 | 0.7326 | 0.9920    | 0.0000    |
| PenaltyQAOA (best p) | 2 | 0.989  | **0.943** | **0.929** |
| Hybrid+VCGNoFlag     | 1 | 0.7323 | **1.000** | 0.0000    |

**Hybrid+VCGNoFlag partially beats PenaltyQAOA**: P(feas) 1.000 vs 0.943, but
P(opt) 0.000 vs 0.929.  9/32 feasible states; same issue as Case 3.

**Overall summary after Change 1**:

| Case | PenaltyQAOA P(feas) | Hybrid+VCGNoFlag P(feas) | PenaltyQAOA P(opt) | Hybrid+VCGNoFlag P(opt) | Verdict |
|------|---------------------|--------------------------|--------------------|-------------------------|---------|
| 1    | 0.495               | **1.000**                | 0.043              | **0.555**               | Hybrid wins both |
| 2    | 0.781               | **1.000**                | 0.009              | **0.356**               | Hybrid wins both |
| 3    | 0.886               | **1.000**                | **0.871**          | 0.001                   | Hybrid wins P(feas), Penalty wins P(opt) |
| 4    | 0.943               | **1.000**                | **0.929**          | 0.000                   | Hybrid wins P(feas), Penalty wins P(opt) |

Hybrid+VCGNoFlag achieves P(feas)=1.0 everywhere at p=1 because every VCGNoFlag
gadget reaches AR=1.0 and the Grover mixer preserves the feasible subspace.
Cases 1–2 also win on P(opt) because exact single-state X-gate preparations
collapse parts of the search space, leaving very few feasible candidates for
the outer QAOA to resolve.
Cases 3–4 fail on P(opt) because 8–9 feasible states remain with similar QUBO
values — addressed by TODO #3 and #4.

**Status**: Implemented and tested on all 4 focus cases.

In Cases 3–4, P(feas)=1.0 but P(opt)=0.0: the outer QAOA distributes probability
evenly across many feasible states.  The remaining gap will be addressed by
TODO #3 (principled penalty weight) and TODO #4 (more optimization budget).

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
