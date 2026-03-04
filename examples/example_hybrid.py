"""
example_hybrid.py -- Toy demo: HybridQAOA vs PenaltyQAOA on a cardinality-constrained QUBO.

Loads the first n=3 QUBO from data/qubos.csv, solves with:
  - HybridQAOA (Dicke state prep + Grover mixer)
  - PenaltyQAOA (penalty-term baseline)

Run from the project root:
    python examples/example_hybrid.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import pennylane as qml
from pennylane import numpy as np

from core import hybrid_qaoa as hq
from core import penalty_qaoa as pq
from core import constraint_handler as ch
from data.make_data import read_qubos_from_file, get_optimal_x


CONSTRAINT = 'x_0 + x_1 + x_2 == 1'
N_X = 3
DATA_DIR = './data/'


def p_feasible(counts: dict, constraints: list, n_x: int) -> float:
    total = sum(counts.values())
    feas = 0
    for bs, cnt in counts.items():
        vd = {f'x_{i}': int(bs[i]) for i in range(n_x)}
        if all(eval(c, {"__builtins__": {}}, vd) for c in constraints):
            feas += cnt
    return feas / total if total > 0 else 0.0


def p_optimal(counts: dict, optimal_bitstrings: list, n_x: int) -> float:
    if optimal_bitstrings is None:
        return float('nan')
    total = sum(counts.values())
    opt = sum(cnt for bs, cnt in counts.items()
              if bs[:n_x] in optimal_bitstrings)
    return opt / total if total > 0 else 0.0


def run_hybrid(Q: np.ndarray, constraints: list) -> dict:
    parsed = ch.parse_constraints(constraints)
    structural_indices = list(range(len(parsed)))

    min_val, optimal_x, total_min = get_optimal_x(Q, constraints)
    penalty = float(5 + 2 * abs(total_min))

    hybrid = hq.HybridQAOA(
        qubo=Q,
        all_constraints=parsed,
        structural_indices=structural_indices,
        penalty_indices=[],
        penalty_str=[penalty],
        angle_strategy='ma-QAOA',
        mixer='Grover',
        n_layers=1,
        steps=50,
        num_restarts=10,
    )
    opt_cost, opt_angles = hybrid.optimize_angles(hybrid.do_evolution_circuit)
    counts = hybrid.do_counts_circuit(shots=10000)

    C_max = max(qml.eigvals(hybrid.problem_ham))
    C_min = min(qml.eigvals(hybrid.problem_ham))
    ar = float((opt_cost - C_max) / (C_min - C_max))

    return {
        'method': 'HybridQAOA',
        'AR': ar,
        'p_feasible': p_feasible(counts, constraints, N_X),
        'p_optimal':  p_optimal(counts, optimal_x, N_X),
    }


def run_penalty(Q: np.ndarray, constraints: list) -> dict:
    min_val, optimal_x, total_min = get_optimal_x(Q, constraints)
    penalty = float(5 + 2 * abs(total_min))

    solver = pq.PenaltyQAOA(
        qubo=Q,
        constraints=constraints,
        penalty=penalty,
        angle_strategy='ma-QAOA',
        n_layers=1,
        steps=50,
        num_restarts=10,
    )
    opt_cost, opt_angles = solver.optimize_angles(solver.do_evolution_circuit)
    counts = solver.do_counts_circuit(shots=10000)

    C_max = max(qml.eigvals(solver.full_Ham))
    C_min = min(qml.eigvals(solver.full_Ham))
    ar = float((opt_cost - C_max) / (C_min - C_max))

    return {
        'method': 'PenaltyQAOA',
        'AR': ar,
        'p_feasible': p_feasible(counts, constraints, N_X),
        'p_optimal':  p_optimal(counts, optimal_x, N_X),
    }


def main() -> None:
    print("Loading QUBOs ...")
    qubos = read_qubos_from_file('qubos.csv', results_dir=DATA_DIR)
    if N_X not in qubos:
        print(f"No QUBO of size {N_X} found in {DATA_DIR}qubos.csv")
        return

    # Use the first QUBO for n=N_X
    qubo_idx = sorted(qubos[N_X].keys())[0]
    Q = qubos[N_X][qubo_idx]['Q']
    qubo_string = qubos[N_X][qubo_idx]['qubo_string']
    constraints = [CONSTRAINT]

    print(f"\nQUBO (n={N_X}): {qubo_string}")
    print(f"Constraint   : {CONSTRAINT}\n")

    print("Running HybridQAOA ...")
    hybrid_res = run_hybrid(Q, constraints)

    print("Running PenaltyQAOA ...")
    penalty_res = run_penalty(Q, constraints)

    print("\n{'Method':<15} {'AR':>8} {'P(feas)':>10} {'P(opt)':>10}")
    print('-' * 48)
    for res in [hybrid_res, penalty_res]:
        print(f"{res['method']:<15} {res['AR']:>8.4f} "
              f"{res['p_feasible']:>10.4f} {res['p_optimal']:>10.4f}")


if __name__ == '__main__':
    main()
