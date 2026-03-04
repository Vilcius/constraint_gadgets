"""
run_utils.py -- Shared helpers for domain-based run scripts.

Provides:
    read_typed_csv(filepath)        -- parse n_vars; ['c1', ...] CSV format
    collect_vcg_data(...)           -- collect metrics from a VCG instance
    collect_pqaoa_data(...)         -- collect metrics from a ProblemQAOA instance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pennylane as qml
from pennylane import numpy as np
from core import vcg as vcg_module
from core import problem_qaoa as pq


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


def collect_pqaoa_data(constraints: list, pqaoa: pq.ProblemQAOA, qubo_string: str,
                       combined: bool = False, overlap: bool = False,
                       single_flag: bool = False, decompose: bool = True,
                       previous_angles=None, min_val=None) -> dict:
    """Collect metrics from a ProblemQAOA instance after optimisation.

    Returns a single-row dict suitable for pd.DataFrame().
    """
    C_max = max(qml.eigvals(pqaoa.problem_Ham))
    C_min = min(qml.eigvals(pqaoa.problem_Ham))
    if pqaoa.n_layers == 1:
        opt_cost, opt_angles = pqaoa.optimize_angles(pqaoa.do_evolution_circuit)
    else:
        opt_cost, opt_angles = pqaoa.optimize_angles(
            pqaoa.do_evolution_circuit, prev_layer_angles=previous_angles
        )
    resources, est_shots, est_error, group_est_shots, group_est_error = pqaoa.get_circuit_resources()
    counts = pqaoa.do_counts_circuit(shots=10000)
    return {
        'qubo_string': [qubo_string],
        'constraints': [constraints],
        'n_x': [pqaoa.n_x],
        'n_c': [len(constraints)],
        'combined': [combined],
        'constraint_penalty': [pqaoa.penalty],
        'overlap': [overlap],
        'overlap_vars': [pqaoa.overlap_vars],
        'overlap_penalty': [pqaoa.overlap_penalty],
        'single_flag': [single_flag],
        'decompose': [decompose],
        'Hamiltonian': [pqaoa.problem_Ham],
        'outcomes': [None],
        'angle_strategy': [pqaoa.angle_strategy],
        'n_layers': [pqaoa.n_layers],
        'num_gamma': [pqaoa.num_gamma],
        'num_beta': [pqaoa.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [opt_cost],
        'counts': [counts],
        'resources': [resources],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [pqaoa.hamiltonian_time],
        'optimize_time': [pqaoa.optimize_time],
        'counts_time': [pqaoa.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'min_val': [min_val],
        'AR': [(opt_cost - C_max) / (C_min - C_max)],
    }
