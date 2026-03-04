"""
generate_params.py -- Generate parameter files for batch Slurm runs.

Produces:
    slurm/vcg_params.txt    -- one line per VCG job
    slurm/hybrid_params.txt -- one line per HybridQAOA job

Line format:
    constraint_type corp n_layers

Run from the project root:
    python slurm/generate_params.py
"""

import os

# Sweeps
CONSTRAINT_TYPES = [
    'cardinality',
    'knapsack',
    'quadratic',
    'flow',
    'assignment',
    'subtour',
]

ANGLE_STRATEGIES = ['QAOA', 'ma-QAOA']
N_LAYERS = [1, 2, 3]
MIXERS = ['Grover', 'X-Mixer']

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def generate_vcg_params(path: str) -> int:
    """Write VCG parameter file: constraint_type corp n_layers."""
    lines = []
    for ct in CONSTRAINT_TYPES:
        for n_layers in N_LAYERS:
            lines.append(f"{ct} constraint {n_layers}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return len(lines)


def generate_hybrid_params(path: str) -> int:
    """Write HybridQAOA parameter file: constraint_type corp n_layers."""
    lines = []
    for ct in CONSTRAINT_TYPES:
        for n_layers in N_LAYERS:
            lines.append(f"{ct} hybrid {n_layers}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return len(lines)


def main() -> None:
    vcg_path    = os.path.join(OUT_DIR, 'vcg_params.txt')
    hybrid_path = os.path.join(OUT_DIR, 'hybrid_params.txt')

    n_vcg    = generate_vcg_params(vcg_path)
    n_hybrid = generate_hybrid_params(hybrid_path)

    print(f"Written {n_vcg} VCG jobs    -> {vcg_path}")
    print(f"Written {n_hybrid} hybrid jobs -> {hybrid_path}")


if __name__ == '__main__':
    main()
