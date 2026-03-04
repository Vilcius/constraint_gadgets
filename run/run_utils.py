"""
run_utils.py -- Shared helpers for domain-based run scripts.

Provides:
    read_typed_csv(filepath)        -- parse n_vars; ['c1', ...] CSV format
    collect_vcg_data(...)           -- collect metrics from a VCG instance
    collect_hybrid_data(...)        -- collect metrics from a HybridQAOA instance
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pennylane as qml
from pennylane import numpy as np
from core import vcg as vcg_module
from core import hybrid_qaoa as hq
from core import constraint_handler as ch


def remap_to_zero_indexed(constraint_str: str, variables: set) -> tuple:
    """Remap variable indices in a constraint string to 0-based.

    Produces a canonical form with x_0, x_1, ... so that VCG training
    uses 0-indexed wires.  The ConstraintMapper in VCG.get_pre_made_data()
    recovers the original variable assignment via permutation search.

    Returns
    -------
    remapped : str
        Constraint string with variables renamed x_0, x_1, ...
    n_c_vars : int
        Number of distinct variables (= flag wire offset for training).
    """
    var_list = sorted(variables)
    var_map = {old: new for new, old in enumerate(var_list)}

    def _replace(m):
        idx = int(m.group(1))
        return f'x_{var_map[idx]}' if idx in var_map else m.group(0)

    return re.sub(r'x_(\d+)', _replace, constraint_str), len(var_list)


def read_typed_csv(filepath: str) -> list:
    """Parse a typed constraint CSV into a list of (n_vars, constraints) tuples.

    Expected line format:
        n_vars; ['constraint_string_1', 'constraint_string_2', ...]

    Returns:
        list of (int, list[str])
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_str, rest = line.split('; ', 1)
            n_vars = int(n_str)
            parsed = eval(rest)
            if isinstance(parsed, str):
                constraints = [parsed]
            elif isinstance(parsed, tuple):
                constraints = list(parsed)
            else:
                constraints = list(parsed)
            rows.append((n_vars, constraints))
    return rows


def collect_vcg_data(vcg: vcg_module.VCG, combined: bool = False,
                     single_flag: bool = False, decompose: bool = True) -> dict:
    """Collect metrics from a VCG instance after optimisation.

    Returns a single-row dict suitable for pd.DataFrame().
    """
    C_max = max(qml.eigvals(vcg.constraint_Ham))
    C_min = min(qml.eigvals(vcg.constraint_Ham))
    opt_cost, opt_angles = vcg.optimize_angles(vcg.do_evolution_circuit)
    resources, est_shots, est_error, group_est_shots, group_est_error = vcg.get_circuit_resources()
    counts = vcg.do_counts_circuit(shots=est_shots)
    return {
        'constraints': [vcg.constraints],
        'n_x': [vcg.n_x],
        'n_c': [len(vcg.constraints)],
        'combined': [combined],
        'single_flag': [single_flag],
        'decompose': [decompose],
        'Hamiltonian': [vcg.constraint_Ham],
        'outcomes': [vcg.outcomes],
        'angle_strategy': [vcg.angle_strategy],
        'n_layers': [vcg.n_layers],
        'num_gamma': [vcg.num_gamma],
        'num_beta': [vcg.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [opt_cost],
        'counts': [counts],
        'resources': [resources],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [vcg.hamiltonian_time],
        'optimize_time': [vcg.optimize_time],
        'counts_time': [vcg.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'AR': [(opt_cost - C_max) / (C_min - C_max)],
    }


def collect_hybrid_data(constraints: list, hybrid: hq.HybridQAOA,
                        qubo_string: str, min_val: float = None,
                        previous_angles=None) -> dict:
    """Collect metrics from a HybridQAOA instance after optimisation.

    Returns a single-row dict suitable for pd.DataFrame().
    """
    C_max = max(qml.eigvals(hybrid.problem_ham))
    C_min = min(qml.eigvals(hybrid.problem_ham))
    if hybrid.n_layers == 1:
        opt_cost, opt_angles = hybrid.optimize_angles(hybrid.do_evolution_circuit)
    else:
        opt_cost, opt_angles = hybrid.optimize_angles(
            hybrid.do_evolution_circuit, prev_layer_angles=previous_angles
        )
    _, est_shots, est_error, group_est_shots, group_est_error = hybrid.get_circuit_resources()
    counts = hybrid.do_counts_circuit(shots=10000)
    return {
        'qubo_string': [qubo_string],
        'constraints': [constraints],
        'n_x': [hybrid.n_x],
        'n_c': [len(constraints)],
        'Hamiltonian': [hybrid.problem_ham],
        'angle_strategy': [hybrid.angle_strategy],
        'mixer': [hybrid.mixer],
        'n_layers': [hybrid.n_layers],
        'num_gamma': [hybrid.num_gamma],
        'num_beta': [hybrid.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [opt_cost],
        'counts': [counts],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [getattr(hybrid, 'hamiltonian_time', None)],
        'optimize_time': [hybrid.optimize_time],
        'counts_time': [hybrid.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'min_val': [min_val],
        'AR': [(opt_cost - C_max) / (C_min - C_max)],
    }
