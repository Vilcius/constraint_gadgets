"""Coefficient-only rules adapted from the preliminary penalty helpers.

Range is sufficient for integer residuals at multiplier >= 1. Maximum-term
and local-change rules are heuristics, not general sufficiency certificates.
"""
import numpy as np


def effective_qubo_coefficients(Q):
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError('Q must be square')
    return np.diag(Q), np.triu(Q + Q.T, k=1)


def coefficient_penalties(Q, constraints):
    for constraint in constraints:
        values = list(constraint.linear.values()) + list(constraint.quadratic.values())
        values += [constraint.constant, constraint.rhs]
        if any(float(v) != round(float(v)) for v in values):
            raise ValueError('These tuning rules require integer-valued constraints')
    linear, upper = effective_qubo_coefficients(Q)
    coefficients = np.concatenate([linear, upper[np.triu_indices(len(linear), 1)]])
    local = np.abs(linear) + np.abs(upper).sum(axis=0) + np.abs(upper).sum(axis=1)
    return dict(range=float(np.abs(coefficients).sum()+1),
                maximum=float(np.abs(coefficients).max()+1),
                local=float(local.max()+1))
