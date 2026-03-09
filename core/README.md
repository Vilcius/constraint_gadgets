# core

Core library for constraint-aware QAOA.

## Modules

| File | Purpose |
|---|---|
| `qaoa_base.py` | Shared primitives: QUBO→Ising conversion, Hamiltonian construction, cost unitaries, X/XY mixers, Adam optimisation loop, resource estimation |
| `constraint_handler.py` | Constraint parsing, type classification, variable sets, slack allocation, feasibility checking, and constraint normalisation for DB lookup |
| `vcg.py` | Variational Constraint Gadget (VCG): builds a Z-diagonal constraint Hamiltonian via Walsh-Hadamard transform and trains a QAOA circuit to prepare the uniform superposition over feasible states |
| `hybrid_qaoa.py` | HybridQAOA: partitions constraints into structural (enforced via VCG/Dicke gadgets) and penalty (added to cost Hamiltonian); runs QAOA in the feasible subspace |
| `penalty_qaoa.py` | PenaltyQAOA: baseline approach that converts all constraints to quadratic penalty terms and runs standard QAOA with X-mixer |
| `dicke_state_prep.py` | Exact Dicke state preparation for `sum x_i == k` constraints (Bartschi & Eidenbenz, 2019); uses XY mixer to preserve Hamming weight |

## Architecture

```
constraint_handler   <── parses / classifies all constraints
       │
       ├── dicke_state_prep  ← exact circuit for sum x_i == k
       │
       └── vcg               ← QAOA-trained gadget for general constraints
              │
              └── qaoa_base  ← shared circuit / optimisation primitives
                     ▲
                     │
            hybrid_qaoa ──── penalty_qaoa
```

`HybridQAOA` and `PenaltyQAOA` both delegate to `qaoa_base` for optimisation.
`HybridQAOA` additionally uses `vcg` and `dicke_state_prep` for its initial state
and Grover mixer.

## Constraint types

| `ConstraintType` | Example | Structural handler |
|---|---|---|
| `DICKE` | `x_0 + x_1 + x_2 == 2` | `DickeStatePrep` + XY mixer |
| `CARDINALITY_LEQ` | `x_0 + x_1 + x_2 <= 2` | `CardinalityLeqStatePrep` + Grover mixer |
| `FLOW` | `x_0 + x_1 - x_2 - x_3 == 0` | `DickeStatePrep` (signed) |
| `WEIGHTED_SUM` | `3*x_0 + 5*x_1 <= 7` | `VCG` |
| `QUADRATIC` | `x_0*x_1 + 2*x_2 <= 1` | `VCG` |

## Angle strategies

| Strategy | Parameters per layer | Description |
|---|---|---|
| `QAOA` | 2 (one γ, one β) | Shared angles across all Pauli terms and qubits |
| `ma-QAOA` | `num_gamma + num_beta` | Independent angle per Pauli term and per qubit |

`ma-QAOA` strictly generalises `QAOA`; QAOA is a special case with all angles
tied equal.
