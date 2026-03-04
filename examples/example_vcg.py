"""
example_vcg.py -- Toy demo: VCG training on a single cardinality constraint.

Run from the project root:
    python examples/example_vcg.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from core import vcg as vcg_module
from analyze_results.plot_utils import setup_style, _ROSE_PINE


CONSTRAINT = 'x_0 + x_1 + x_2 == 1'   # 3-variable cardinality


def run_vcg(angle_strategy: str) -> dict:
    n_x = 3
    flag_wires = [n_x]   # one flag qubit
    gadget = vcg_module.VCG(
        constraints=[CONSTRAINT],
        flag_wires=flag_wires,
        angle_strategy=angle_strategy,
        n_layers=1,
        steps=50,
        num_restarts=20,
    )
    opt_cost, opt_angles = gadget.optimize_angles(gadget.do_evolution_circuit)
    resources, est_shots, _, _, _ = gadget.get_circuit_resources()
    counts = gadget.do_counts_circuit(shots=est_shots)

    # P(feasible): flag bit == '0'
    total = sum(counts.values())
    feasible = sum(v for k, v in counts.items() if k[-1] == '0')
    p_feas = feasible / total if total > 0 else 0.0

    # AR
    import pennylane as qml
    eigvals = qml.eigvals(gadget.constraint_Ham)
    C_max, C_min = max(eigvals), min(eigvals)
    ar = float((opt_cost - C_max) / (C_min - C_max))

    return {
        'angle_strategy': angle_strategy,
        'AR': ar,
        'p_feasible': p_feas,
        'depth': resources.depth,
        'num_gates': resources.num_gates,
        'counts': counts,
    }


def plot_counts(results: list[dict]) -> None:
    setup_style()
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    colors = [_ROSE_PINE['gold'], _ROSE_PINE['pine']]

    for ax, res, color in zip(axes, results, colors):
        counts = res['counts']
        keys = sorted(counts.keys())
        vals = [counts[k] for k in keys]
        total = sum(vals)
        probs = [v / total for v in vals]

        ax.bar(keys, probs, color=color, alpha=0.8)
        ax.set_title(res['angle_strategy'])
        ax.set_xlabel('Bitstring')
        ax.set_ylabel('Probability')
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(f'VCG measurement distribution\n{CONSTRAINT}')
    plt.tight_layout()
    plt.savefig('examples/vcg_example_counts.png', dpi=120)
    plt.show()


def main() -> None:
    print(f"Constraint: {CONSTRAINT}\n")
    results = []
    for strategy in ['QAOA', 'ma-QAOA']:
        print(f"Training VCG [{strategy}] ...")
        res = run_vcg(strategy)
        results.append(res)
        print(f"  AR          = {res['AR']:.4f}")
        print(f"  P(feasible) = {res['p_feasible']:.4f}")
        print(f"  Depth       = {res['depth']}")
        print(f"  Num gates   = {res['num_gates']}")
        print()

    plot_counts(results)


if __name__ == '__main__':
    main()
