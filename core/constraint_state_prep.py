"""Exact local feasible-state preparation with PennyLane StatePrep."""
from functools import lru_cache

import numpy as np
import pennylane as qml

from . import constraint_handler as ch


@lru_cache(maxsize=128)
def _local_preparation(constraint):
    pc = ch.parse_constraints([constraint])[0]
    wires = tuple(sorted(pc.variables))
    if not wires:
        raise ValueError("StatePrep requires a nonempty constraint support")
    mask = []
    for index in range(1 << len(wires)):
        bits = format(index, f"0{len(wires)}b")
        assignment = ["0"] * (max(wires) + 1)
        for wire, bit in zip(wires, bits):
            assignment[wire] = bit
        mask.append(ch.check_feasibility("".join(assignment), [pc]))
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError(f"Constraint has no feasible assignments: {constraint}")
    state = mask.astype(float) / np.sqrt(mask.sum())
    # Decompose StatePrep once with fixed numerical amplitudes.  This gives
    # a unitary gate circuit usable both forwards and as an adjoint in the
    # existing Grover mixer, and avoids amplitude preprocessing under JAX.
    with qml.QueuingManager.stop_recording():
        preparation = qml.StatePrep(state, wires=range(len(wires)))
        pending = list(preparation.decomposition())
        operations = []
        while pending:
            operation = pending.pop(0)
            if operation.name in ("RY", "RZ", "CNOT", "GlobalPhase"):
                operations.append(operation)
            else:
                pending[0:0] = operation.decomposition()
    return wires, state, mask, tuple(operations)


class ConstraintStatePrep:
    """The same preparation/mixer interface as a VCG, without training."""

    def __init__(self, constraint):
        wires, state, mask, local_operations = _local_preparation(constraint)
        self.var_wires = list(wires)
        self.state = state.copy()
        self.feasible_mask = mask.copy()
        self.constraint = constraint
        mapping = dict(enumerate(wires))
        with qml.QueuingManager.stop_recording():
            self.operations = tuple(op.map_wires(mapping) for op in local_operations)

    def opt_circuit(self):
        for operation in self.operations:
            qml.apply(operation)

    def mixer_circuit(self, beta):
        from . import qaoa_base as base
        base.apply_grover_mixer(beta, self.var_wires, [self])
