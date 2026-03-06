"""
qaoa_base.py -- Shared QAOA / Hamiltonian utilities.

Consolidates functionality duplicated across VCG, ProblemQAOA,
PenaltyQAOA, and HybridQAOA:

  - QUBO <-> Ising conversion
  - Hamiltonian construction (QUBO, flag-penalty, overlap)
  - Cost unitary and mixer application
  - Angle initialisation (QAOA / ma-QAOA, warm-start)
  - Optimisation loop (Adam with random restarts, convergence)
  - Resource estimation (shots, grouping)
  - Validation helpers
"""

from __future__ import annotations

import re
import time
import itertools as it
from typing import Tuple, List, Optional

import pennylane as qml
from pennylane import numpy as np


# ======================================================================
# Ising / Hamiltonian utilities (stateless, module-level)
# ======================================================================

def qubo_to_ising(
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert an upper-triangular QUBO matrix to Ising coefficients.

    Given  f(x) = x^T Q x  with x in {0,1}^n, substituting
    x_i = (1 - z_i)/2  yields:

        f(z) = sum_{i<j} J_{ij} z_i z_j  +  sum_i h_i z_i  +  offset

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        QUBO coefficient matrix.

    Returns
    -------
    J : np.ndarray, shape (n, n)
        Quadratic Ising coefficients (upper-triangular).
    h : np.ndarray, shape (n,)
        Linear Ising coefficients.
    offset : float
        Constant energy offset.
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    diag = np.diag(Q)

    for i in range(n):
        h[i] -= (
            0.5 * diag[i]
            + 0.25 * sum(Q[k][i] for k in range(i))
            + 0.25 * sum(Q[i][l] for l in range(i + 1, n))
        )
        for j in range(i + 1, n):
            J[i][j] += Q[i][j] / 4

    offset = float(np.sum(Q) / 4 + np.sum(diag) / 4)
    return J, h, offset


def build_qubo_hamiltonian(
    Q: np.ndarray,
    wires: List[int],
) -> qml.Hamiltonian:
    """
    Build the Ising-form Hamiltonian corresponding to a QUBO matrix.

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix (n x n).
    wires : list[int]
        Qubit wire indices for the n decision variables.

    Returns
    -------
    qml.Hamiltonian
    """
    n = Q.shape[0]
    J, h, offset = qubo_to_ising(Q)

    coeffs, obs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                coeffs.append(J[i, j])
                obs.append(qml.PauliZ(wires[i]) @ qml.PauliZ(wires[j]))
        if h[i] != 0:
            coeffs.append(h[i])
            obs.append(qml.PauliZ(wires[i]))

    if offset != 0:
        coeffs.append(offset)
        obs.append(qml.Identity(wires[0]))

    return qml.Hamiltonian(np.array(coeffs), obs)


def build_flag_penalty_hamiltonian(
    flag_wires: List[int],
    penalties: List[float],
) -> Optional[qml.Hamiltonian]:
    """
    Build  sum_k (pen_k / 2)(I - Z_{flag_k})  for structural flag qubits.

    A flag qubit in |1> (Z eigenvalue -1) adds cost pen_k, biasing the
    optimiser toward |0> (feasible).

    Returns
    -------
    qml.Hamiltonian or None (if flag_wires is empty).
    """
    if not flag_wires:
        return None
    coeffs, obs = [], []
    for fw, pen in zip(flag_wires, penalties):
        coeffs += [pen / 2, -pen / 2]
        obs += [qml.Identity(fw), qml.PauliZ(fw)]
    return qml.Hamiltonian(coeffs, obs)



# ======================================================================
# Circuit building blocks
# ======================================================================

def apply_cost_unitary(
    hamiltonian: qml.Hamiltonian,
    gammas: np.ndarray,
    layer: int,
) -> None:
    """
    Apply exp(-i gamma_k H_k) for each non-identity Pauli term in the Hamiltonian.

    Parameters
    ----------
    hamiltonian : qml.Hamiltonian
        Cost Hamiltonian (Pauli decomposition).
    gammas : np.ndarray
        2-D array of gamma angles, shape (n_layers, num_gamma).
    layer : int
        Current QAOA layer index.
    """
    idx = 0
    coeffs, ops = hamiltonian.terms()
    for w, op in zip(coeffs, ops):
        if re.search(r"^[I]+$", qml.pauli.pauli_word_to_string(op)):
            continue
        qml.MultiRZ(w * gammas[layer][idx], wires=op.wires)
        idx += 1


def apply_x_mixer(
    betas: np.ndarray,
    layer: int,
    wires: List[int],
) -> None:
    """
    Standard X-mixer: product_i RX(beta_i, wire_i).
    """
    for i, wire in enumerate(wires):
        qml.RX(betas[layer][i], wires=wire)


def apply_grover_mixer(
    beta: float,
    all_wires: List[int],
    state_prep_circuits: list,
) -> None:
    """
    Grover diffusion mixer:  A^dag (2|s><s| - I) A

    where A is the composite state preparation from the gadget circuits.

    Parameters
    ----------
    beta : float
        Single mixer angle.
    all_wires : list[int]
        All qubit wires in the circuit.
    state_prep_circuits : list
        Objects with an ``opt_circuit()`` method (e.g. VCG, DickeStatePrep).
    """
    # Un-prepare
    for gadget in reversed(state_prep_circuits):
        qml.adjoint(gadget.opt_circuit)()

    # Reflection about |0...0>
    for wire in all_wires:
        qml.PauliX(wires=wire)
    qml.ctrl(
        qml.PhaseShift(beta / np.pi, wires=all_wires[-1]),
        control=all_wires[:-1],
    )
    for wire in all_wires:
        qml.PauliX(wires=wire)

    # Re-prepare
    for gadget in state_prep_circuits:
        gadget.opt_circuit()


def apply_xy_mixer(
    beta: float,
    wires: List[int],
    ring: bool = True,
) -> None:
    """
    XY-mixer (Hamming-weight preserving).

    Applies partial-SWAP-like interactions:
        exp(-i beta (X_i X_j + Y_i Y_j) / 2)
    on neighbouring pairs.

    Parameters
    ----------
    beta : float
        Mixer angle.
    wires : list[int]
        Qubit wires to mix over.
    ring : bool
        If True, also couple the last wire to the first (Ring-XY).
    """
    n = len(wires)
    # Even bonds
    for k in range(0, n - 1, 2):
        _apply_xy_interaction(beta, wires[k], wires[k + 1])
    # Odd bonds
    for k in range(1, n - 1, 2):
        _apply_xy_interaction(beta, wires[k], wires[k + 1])
    # Ring closure
    if ring and n > 2:
        _apply_xy_interaction(beta, wires[-1], wires[0])


def _apply_xy_interaction(beta: float, wire_a: int, wire_b: int) -> None:
    """Single XY interaction: exp(-i beta (XX + YY)/2) on two qubits."""
    qml.CNOT(wires=[wire_a, wire_b])
    qml.RY(beta, wires=wire_a)
    qml.RY(beta, wires=wire_b)
    qml.CNOT(wires=[wire_a, wire_b])


# ======================================================================
# Angle utilities
# ======================================================================

def init_angles(
    n_layers: int,
    num_gamma: int,
    num_beta: int,
    angle_strategy: str,
    prev_layer_angles: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate initial QAOA angles for one optimisation restart.

    Returns
    -------
    np.ndarray
        Shape (n_layers, num_gamma + num_beta) for ma-QAOA,
        or    (n_layers, 2) for standard QAOA.
    """
    if angle_strategy == "ma-QAOA":
        total = num_gamma + num_beta
        if prev_layer_angles is not None:
            new_size = n_layers * total - prev_layer_angles.size
            new = np.random.uniform(
                -2 * np.pi, 2 * np.pi, (new_size,), requires_grad=True
            )
            return np.concatenate(
                [prev_layer_angles.flatten(), new]
            ).reshape(n_layers, total)
        return np.random.uniform(
            -2 * np.pi, 2 * np.pi, n_layers * total, requires_grad=True
        ).reshape(n_layers, total)
    else:
        if prev_layer_angles is not None:
            new_size = 2 * n_layers - prev_layer_angles.size
            new = np.random.uniform(
                -2 * np.pi, 2 * np.pi, (new_size,), requires_grad=True
            )
            return np.concatenate(
                [prev_layer_angles.flatten(), new]
            ).reshape(n_layers, 2)
        return np.random.uniform(
            -2 * np.pi, 2 * np.pi, 2 * n_layers, requires_grad=True
        ).reshape(n_layers, 2)


def split_angles(
    angles: np.ndarray,
    num_gamma: int,
    num_beta: int,
    angle_strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a parameter array into (gammas, betas).

    For ma-QAOA:  gammas = angles[:, :num_gamma], betas = angles[:, num_gamma:]
    For QAOA:     values are broadcast to full width.
    """
    if angle_strategy == "ma-QAOA":
        return angles[:, :num_gamma], angles[:, num_gamma:]
    else:
        gammas = np.array([angles[:, 0]] * num_gamma).T
        betas = np.array([angles[:, 1]] * num_beta).T
        return gammas, betas


def convert_qaoa_to_ma_angles(
    angles: np.ndarray,
    num_gamma: int,
    num_beta: int,
    n_layers: int,
) -> np.ndarray:
    """
    Broadcast standard QAOA angles (one gamma, one beta per layer) into
    the ma-QAOA shape by repeating each value.
    """
    ma = []
    for layer_idx in range(n_layers):
        g = angles[layer_idx, 0]
        b = angles[layer_idx, 1]
        ma += [g] * num_gamma + [b] * num_beta
    return np.array(ma).reshape(n_layers, num_gamma + num_beta)


# ======================================================================
# Optimisation
# ======================================================================

def run_optimization(
    cost_fn,
    n_layers: int,
    num_gamma: int,
    num_beta: int,
    angle_strategy: str,
    steps: int = 50,
    num_restarts: int = 10,
    learning_rate: float = 0.1,
    conv_tol: float = 1e-6,
    maximize: bool = False,
    prev_layer_angles: Optional[np.ndarray] = None,
    starting_angles: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, float]:
    """
    Optimise QAOA angles using Adam with random restarts.

    Parameters
    ----------
    cost_fn : callable
        Takes angles array, returns scalar expectation value.
    n_layers, num_gamma, num_beta : int
        Circuit dimensioning.
    angle_strategy : str
        "ma-QAOA" or "QAOA".
    steps : int
        Max gradient steps per restart.
    num_restarts : int
        Number of independent random restarts.
    learning_rate : float
        Adam step size.
    conv_tol : float
        Early stopping tolerance on cost change.
    maximize : bool
        If True, negate cost for maximisation.
    prev_layer_angles : np.ndarray or None
        Warm-start from a shallower circuit.
    starting_angles : np.ndarray or None
        Fixed starting point (used for first restart only).

    Returns
    -------
    best_cost : float
    best_angles : np.ndarray
    wall_time : float
    """
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    opt_mult = -1 if maximize else 1

    best_cost = float("inf")
    best_angles = None

    start = time.time()
    for restart_idx in range(num_restarts):
        if starting_angles is not None and restart_idx == 0:
            angles = starting_angles.copy()
        else:
            angles = init_angles(
                n_layers, num_gamma, num_beta, angle_strategy, prev_layer_angles
            )

        new_cost = opt_mult * cost_fn(angles)
        for _ in range(steps):
            angles, prev_cost = opt.step_and_cost(cost_fn, angles)
            new_cost = cost_fn(angles)
            if np.abs(new_cost - prev_cost) <= conv_tol:
                break

        if new_cost < best_cost:
            best_cost = new_cost
            best_angles = angles

    wall_time = time.time() - start
    return best_cost, best_angles, wall_time


# ======================================================================
# Resource estimation
# ======================================================================

def estimate_hamiltonian_resources(
    hamiltonian: qml.Hamiltonian,
) -> Tuple[float, float, float, float]:
    """
    Estimate shot budget and statistical error for a Pauli-decomposed Hamiltonian.

    Returns
    -------
    est_shots, est_error, group_shots, group_error
    """
    coeffs, ops = hamiltonian.terms()
    est_shots = qml.resource.estimate_shots(coeffs)
    est_error = qml.resource.estimate_error(coeffs)
    group_ops, group_coeffs = qml.pauli.group_observables(ops, coeffs)
    group_coeffs_arr = [np.array(gc) for gc in group_coeffs]
    group_shots = qml.resource.estimate_shots(group_coeffs_arr)
    group_error = qml.resource.estimate_error(group_coeffs_arr)
    return est_shots, est_error, group_shots, group_error


# ======================================================================
# Counting helpers
# ======================================================================

def count_gamma_terms(hamiltonian: qml.Hamiltonian) -> int:
    """Count non-identity Pauli terms (= number of gamma params for ma-QAOA)."""
    _, ops = hamiltonian.terms()
    return sum(
        1 for op in ops
        if not re.search(r"^[I]+$", qml.pauli.pauli_word_to_string(op))
    )


# ======================================================================
# Validation
# ======================================================================

def validate_angle_strategy(s: str) -> str:
    if s in ("QAOA", "ma-QAOA"):
        return s
    raise ValueError("angle_strategy must be 'QAOA' or 'ma-QAOA'.")


def validate_mixer(m: str) -> str:
    valid = ("Grover", "X-Mixer", "XY", "Ring-XY", "SWAP")
    if m in valid:
        return m
    raise ValueError(f"mixer must be one of {valid}.")


# ======================================================================
# Penalty Hamiltonian helpers (Pauli expansion via x_i = (I - Z_i)/2)
# ======================================================================

def _add_squared_term(
    coeffs: list, obs: list, delta: float, coeff: float, wires: list
) -> None:
    """Add delta * coeff^2 * (product x_i)^2 = delta * coeff^2 * product x_i."""
    if len(wires) == 1:
        c = delta * coeff ** 2
        coeffs += [c * 0.5, -c * 0.5]
        obs += [qml.Identity(wires[0]), qml.PauliZ(wires[0])]
    elif len(wires) == 2:
        c = delta * coeff ** 2
        coeffs += [c * 0.25, -c * 0.25, -c * 0.25, c * 0.25]
        obs += [
            qml.Identity(wires[0]),
            qml.PauliZ(wires[0]),
            qml.PauliZ(wires[1]),
            qml.PauliZ(wires[0]) @ qml.PauliZ(wires[1]),
        ]


def _add_two_var_product(
    coeffs: list, obs: list, coeff: float, w_a: int, w_b: int
) -> None:
    """Add coeff * x_a * x_b = coeff * (I - Z_a - Z_b + Z_a Z_b) / 4."""
    coeffs += [coeff * 0.25, -coeff * 0.25, -coeff * 0.25, coeff * 0.25]
    obs += [
        qml.Identity(w_a),
        qml.PauliZ(w_a),
        qml.PauliZ(w_b),
        qml.PauliZ(w_a) @ qml.PauliZ(w_b),
    ]


def _add_multiway_product(
    coeffs: list, obs: list, coeff: float, wires: list
) -> None:
    """
    Expand coeff * product_i x_i via x_i = (I - Z_i)/2.

    Binary variables satisfy x_i^2 = x_i, so duplicate wires are collapsed.
    """
    unique_wires = list(set(wires))
    n = len(unique_wires)
    for size in range(n + 1):
        for subset in it.combinations(unique_wires, size):
            term_coeff = coeff / (2 ** n) * ((-1) ** size)
            if len(subset) == 0:
                coeffs.append(term_coeff)
                obs.append(qml.Identity(unique_wires[0]))
            elif len(subset) == 1:
                coeffs.append(term_coeff)
                obs.append(qml.PauliZ(subset[0]))
            else:
                coeffs.append(term_coeff)
                pauli = qml.PauliZ(subset[0])
                for w in subset[1:]:
                    pauli = pauli @ qml.PauliZ(w)
                obs.append(pauli)


def build_penalty_hamiltonian(
    pen_constraints,
    slack_infos,
    delta: float,
    fallback_wire: int = 0,
) -> qml.Hamiltonian:
    """
    Build  delta * sum_k (c_k(x, s) - b_k)^2  for penalised constraints.

    Expands each squared constraint expression into Pauli-Z terms via
    the substitution x_i = (I - Z_i)/2, including slack variables.

    Parameters
    ----------
    pen_constraints : list[ParsedConstraint]
        Penalised constraints (from constraint_handler).
    slack_infos : list[SlackInfo]
        Slack variable info for each constraint (from constraint_handler).
    delta : float
        Penalty weight.
    fallback_wire : int
        Wire index for constant-term Identity operators.

    Returns
    -------
    qml.Hamiltonian
    """
    pen_coeffs: List[float] = []
    pen_obs: List = []

    for slack_info in slack_infos:
        pc = pen_constraints[slack_info.constraint_idx]
        rhs = slack_info.effective_rhs

        # Collect (coefficient, [wire_indices]) for each LHS term
        terms: List[Tuple[float, List[int]]] = []
        for var_idx, coeff in pc.linear.items():
            terms.append((coeff, [var_idx]))
        for var_pair, coeff in pc.quadratic.items():
            terms.append((coeff, list(var_pair)))

        # Slack variables
        if slack_info.n_slack > 0:
            sign = 1.0 if slack_info.operator == "leq" else -1.0
            for sw in range(
                slack_info.slack_start_wire,
                slack_info.slack_start_wire + slack_info.n_slack,
            ):
                terms.append((sign, [sw]))

        constant_term = pc.constant - rhs

        # 1. Squared terms: (c_i * term_i)^2
        for coeff, wires in terms:
            _add_squared_term(pen_coeffs, pen_obs, delta, coeff, wires)

        # 2. Cross terms: 2 * c_i * c_j * term_i * term_j
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                ci, wi = terms[i]
                cj, wj = terms[j]
                cross = 2 * ci * cj
                if len(wi) == 1 and len(wj) == 1:
                    _add_two_var_product(pen_coeffs, pen_obs, delta * cross, wi[0], wj[0])
                else:
                    _add_multiway_product(pen_coeffs, pen_obs, delta * cross, wi + wj)

        # 3. Constant-linear interaction: 2 * constant * c_i * term_i
        if constant_term != 0:
            for coeff, wires in terms:
                factor = 2 * constant_term * coeff
                if len(wires) == 1:
                    pen_coeffs += [delta * factor * 0.5, -delta * factor * 0.5]
                    pen_obs += [qml.Identity(wires[0]), qml.PauliZ(wires[0])]
                elif len(wires) == 2:
                    _add_two_var_product(pen_coeffs, pen_obs, delta * factor, wires[0], wires[1])

        # 4. Constant squared: constant^2
        if constant_term != 0:
            pen_coeffs.append(delta * constant_term ** 2)
            pen_obs.append(qml.Identity(fallback_wire))

    return qml.Hamiltonian(pen_coeffs, pen_obs)
