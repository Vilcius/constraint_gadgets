# VCG (flag) vs VCG — Comparison

## Problem setup (example_hybrid.py)

- 7 decision variables
- Constraint A: `x_0 + x_1 + x_2 == 1`  →  Dicke state (structural, no gadget)
- Constraint B: `2*x_3 + 1*x_4 + 4*x_5 <= 2`  →  **gadget trained here**
- Constraint C: `x_1 + x_4 + x_6 <= 1`  →  penalized in cost Hamiltonian
- Optimal bitstring: `1000100`, QUBO value: −11.0

---

## 1. Standalone gadget comparison (Constraint B only)

Both gadgets are trained independently on Constraint B (`n_x = 3` decision variables).

| Gadget       | Qubits    | AR     | P(feas) method   | P(feas) | Layers |
|--------------|-----------|--------|------------------|---------|--------|
| VCG (flag)   | n_x + 1 = 4 | 0.8062 | flag bit = 0     | 0.4634  | 1      |
| VCG    | n_x = 3   | 1.0000 | constraint check | 1.0000  | 1      |

**VCG wins on every metric at the same depth.**

Key differences:

- **AR**: VCG achieves the ground state exactly (AR = 1.0); VCG (flag) is stuck at
  0.8062 with the same training budget.
- **P(feas)**: VCG's P(feas) = 1.0 by direct constraint evaluation. VCG (flag)
  reports P(feas) via the flag bit, which is a conservative proxy — flag=1 does not
  always mean infeasible, so 0.4634 is an underestimate of the true feasible fraction.
- **Qubits**: VCG needs one fewer qubit (no ancilla flag qubit).
- **Hamiltonian**: VCG (flag) operates on n_x + 1 qubits (2^(n_x+1) states, half
  "impossible" flag assignments). VCG operates directly on n_x qubits (2^n_x
  states), making the Hamiltonian smaller and the optimisation landscape simpler.

---

## 2. Full solver results (HybridQAOA uses VCG flag-based for Constraint B)

VCG has not yet been run inside HybridQAOA.  The current results use the
flag-based VCG as the state-prep gadget for Constraint B.

| Method       | AR     | P(feasible) | P(optimal) | Notes                          |
|--------------|--------|-------------|------------|--------------------------------|
| HybridQAOA   | 0.8880 | 0.3040      | 0.0040     | VCG(flag) for B; penalty for C |
| PenaltyQAOA  | 0.9358 | 0.4832      | 0.0024     | AR inflated by penalty (~3000–4000× C_max) |

> **Warning:** AR is NOT comparable between methods. PenaltyQAOA inflates C_max via
> penalty terms (~3000–4000) vs HybridQAOA (~60–100), making its AR appear
> artificially higher. P(feasible) and P(optimal) are the fair metrics.

---

## 3. What VCG inside HybridQAOA would look like

`VCG` exposes `opt_circuit()` and `flag_wires = []`, so it is compatible
with HybridQAOA's Grover mixer interface.  The effect would be:

- No flag qubit added to the problem circuit for Constraint B.
- No flag-penalty term added to the cost Hamiltonian for Constraint B
  (since `needs_flag_penalty = False`).
- The prepared state has P(feas) = 1.0 for Constraint B from the start, vs
  P(feas) ≈ 0.46 with the flag-based VCG.

Expected improvement: since VCG starts from a perfect feasible superposition
for Constraint B, the outer QAOA should need fewer layers to find the optimal
solution.  P(feasible) and P(optimal) for the full problem should improve,
especially for constraints where VCG achieves AR = 1.0.

---

## 4. Open questions / next steps

1. **Run HybridQAOA with VCG** for Constraint B and compare P(feas) and
   P(opt) against the flag-based baseline.
2. **Does VCG always reach AR = 1.0?**  The Constraint B case (3 variables)
   is small.  For larger n_x, check whether VCG still converges at p=1.
3. **Codebase cleanup**: once the full-solver comparison is done, decide whether to:
   - `core/vcg.py` is now the canonical flag-free VCG gadget, or
   - Keep both and let `HybridQAOA` accept either via duck typing.
4. **Feasibility-conditioned AR**: compute AR over feasible shots only to make
   HybridQAOA and PenaltyQAOA directly comparable (see TODO.md item 1).
