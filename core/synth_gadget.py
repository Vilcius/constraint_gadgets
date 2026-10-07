"""
synth_gadget.py -- Exact constraint gadgets by state-preparation synthesis.

A constraint gadget only has to satisfy U|0...0> = |F>, the uniform
superposition over the feasible assignments of the constraint, on the
constraint's support.  Instead of training a QAOA circuit towards |F>, we
write |F> down from the truth table and hand it to a state-preparation
synthesiser.  Several back-ends are tried; each result is transpiled to
{cx, u} and the circuit with the fewest CNOTs is kept.

Bit ordering
------------
Amplitude index i corresponds to the bitstring of the constraint's sorted
variables with x_{v[0]} as the most significant bit (PennyLane / truth-table
convention used everywhere in this repo).  Qiskit is little-endian, so qiskit
qubit q holds variable v[n-1-q]; with that mapping the integer index of each
basis state is identical in both frameworks and the amplitude vector needs
no reordering.
"""

from __future__ import annotations

import itertools as it
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Isometry, StatePreparation
from qiskit.quantum_info import Statevector

from . import constraint_handler as ch

BASIS_GATES = ["cx", "u"]


# ──────────────────────────────────────────────────────────────────────────
# Target state
# ──────────────────────────────────────────────────────────────────────────

def feasible_mask(constraint: str) -> tuple[list[int], np.ndarray]:
    """Return (sorted support variables, boolean feasibility mask over 2^s)."""
    parsed = ch.parse_constraint(constraint)
    wires = sorted(parsed.variables)
    pos = {w: i for i, w in enumerate(wires)}
    mask = []
    for bits in it.product([0, 1], repeat=len(wires)):
        lhs = parsed.constant
        lhs += sum(c * bits[pos[v]] for v, c in parsed.linear.items())
        lhs += sum(c * bits[pos[i]] * bits[pos[j]]
                   for (i, j), c in parsed.quadratic.items())
        mask.append(_holds(lhs, parsed.op, parsed.rhs))
    return wires, np.array(mask, dtype=bool)


def _holds(lhs: float, op: ch.ConstraintOp, rhs: float) -> bool:
    if op == ch.ConstraintOp.EQ:
        return abs(lhs - rhs) < 1e-6
    if op == ch.ConstraintOp.LEQ:
        return lhs <= rhs + 1e-6
    if op == ch.ConstraintOp.LT:
        return lhs < rhs - 1e-6
    if op == ch.ConstraintOp.GEQ:
        return lhs >= rhs - 1e-6
    if op == ch.ConstraintOp.GT:
        return lhs > rhs + 1e-6
    raise ValueError(f"Unknown operator {op}")


def target_state(mask: np.ndarray) -> np.ndarray:
    """Uniform superposition over the feasible indices."""
    if not mask.any():
        raise ValueError("Constraint has no feasible assignment")
    return mask.astype(float) / np.sqrt(mask.sum())


# ──────────────────────────────────────────────────────────────────────────
# Synthesis back-ends: amplitude vector -> QuantumCircuit
# ──────────────────────────────────────────────────────────────────────────

def _qiskit_stateprep(vec: np.ndarray) -> QuantumCircuit:
    n = int(np.log2(len(vec)))
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(vec), range(n))
    return qc


def _qiskit_isometry(vec: np.ndarray) -> QuantumCircuit:
    n = int(np.log2(len(vec)))
    qc = QuantumCircuit(n)
    qc.append(Isometry(vec.reshape(-1, 1), 0, 0), range(n))
    return qc


SYNTHESISERS: Dict[str, Callable[[np.ndarray], QuantumCircuit]] = {
    "qiskit_stateprep": _qiskit_stateprep,
    "qiskit_isometry": _qiskit_isometry,
}


def _register_qclib() -> None:
    """Add qclib back-ends when the package is installed."""
    try:
        from qclib.state_preparation import (
            CvoqramInitialize, LowRankInitialize, MergeInitialize,
            PivotInitialize, UCGEInitialize, UCGInitialize,
        )
    except ImportError:
        return

    def dense(cls, **opts):
        def build(vec):
            return cls(vec, opt_params=opts or None).definition
        return build

    def sparse(cls, **opts):
        def build(vec):
            n = int(np.log2(len(vec)))
            data = {format(i, f"0{n}b"): a for i, a in enumerate(vec) if abs(a) > 1e-12}
            if cls is MergeInitialize:
                # Takes no opt_params and reads keys little-endian.
                return cls({k[::-1]: a for k, a in data.items()}).definition
            return cls(data, opt_params=opts or None).definition
        return build

    SYNTHESISERS.update({
        "qclib_lowrank": dense(LowRankInitialize),
        "qclib_ucg": dense(UCGInitialize),
        "qclib_ucge": dense(UCGEInitialize),
        "qclib_cvoqram": sparse(CvoqramInitialize),
        "qclib_pivot": sparse(PivotInitialize),
        "qclib_merge": sparse(MergeInitialize),     # Gleinig & Hoefler
    })


_register_qclib()


# ──────────────────────────────────────────────────────────────────────────
# Compilation and metrics
# ──────────────────────────────────────────────────────────────────────────

def circuit_metrics(qc: QuantumCircuit) -> dict:
    ops = qc.count_ops()
    return {
        "cx": int(ops.get("cx", 0)),
        "gates": int(sum(ops.values())),
        "depth": int(qc.depth()),
        "cx_depth": int(qc.depth(lambda inst: inst.operation.num_qubits == 2)),
    }


def compile_min_cx(qc: QuantumCircuit, seeds=range(5)) -> QuantumCircuit:
    """Transpile to {cx, u} at optimization level 3; keep fewest CNOTs."""
    best = None
    for seed in seeds:
        out = transpile(qc, basis_gates=BASIS_GATES, optimization_level=3,
                        seed_transpiler=seed)
        key = (out.count_ops().get("cx", 0), out.depth())
        if best is None or key < best[0]:
            best = (key, out)
    return best[1]


def fidelity(qc: QuantumCircuit, vec: np.ndarray) -> float:
    psi = Statevector(qc).data
    return float(abs(np.vdot(vec, psi)) ** 2)


# ──────────────────────────────────────────────────────────────────────────
# Gadget
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SynthGadget:
    """
    Exact state-preparation gadget for one constraint.

    Exposes the same opt_circuit() / mixer_circuit() interface as VCG and
    the Dicke preps, so PC-QAOA can use it as a drop-in structural gadget.
    The circuit is stored as a {cx, u} gate list on qiskit qubit indices.
    """

    constraint: str
    var_wires: List[int]
    n_feasible: int
    method: str
    gates: List[tuple]              # (name, params, qiskit qubits)
    metrics: dict
    fidelity: float
    candidates: Dict[str, dict] = field(default_factory=dict)

    @property
    def n_x(self) -> int:
        return len(self.var_wires)

    def _wire(self, q: int) -> int:
        return self.var_wires[self.n_x - 1 - q]

    def opt_circuit(self) -> None:
        for name, params, qubits in self.gates:
            wires = [self._wire(q) for q in qubits]
            if name == "u":
                qml.U3(*params, wires=wires[0])
            elif name == "cx":
                qml.CNOT(wires=wires)
            else:
                raise ValueError(f"Unexpected gate {name}")

    def mixer_circuit(self, beta: float) -> None:
        """Per-gadget Grover mixer on this gadget's qubits."""
        from . import qaoa_base as base
        base.apply_grover_mixer(beta, self.var_wires, [self])


def _gate_list(qc: QuantumCircuit) -> List[tuple]:
    return [(inst.operation.name, [float(p) for p in inst.operation.params],
             [qc.find_bit(q).index for q in inst.qubits]) for inst in qc.data]


def synthesize(constraint: str, methods: Optional[List[str]] = None,
               seeds=range(5), fidelity_tol: float = 1e-8) -> SynthGadget:
    """Try every back-end, keep the exact circuit with the fewest CNOTs."""
    wires, mask = feasible_mask(constraint)
    vec = target_state(mask)
    candidates, best = {}, None
    for name in methods or list(SYNTHESISERS):
        start = time.time()
        try:
            qc = SYNTHESISERS[name](vec)
            if qc.num_qubits != len(wires):
                raise ValueError(f"uses {qc.num_qubits - len(wires)} ancilla qubits")
            qc = compile_min_cx(qc, seeds)
        except Exception as exc:  # a back-end can reject a given input
            candidates[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        info = circuit_metrics(qc)
        info["fidelity"] = fidelity(qc, vec)
        info["seconds"] = time.time() - start
        candidates[name] = info
        if info["fidelity"] < 1 - fidelity_tol:
            continue
        key = (info["cx"], info["depth"])
        if best is None or key < best[0]:
            best = (key, name, qc, info)
    if best is None:
        raise RuntimeError(f"No exact synthesis for {constraint}: {candidates}")
    _, name, qc, info = best
    return SynthGadget(
        constraint=constraint, var_wires=wires, n_feasible=int(mask.sum()),
        method=name, gates=_gate_list(qc),
        metrics={k: info[k] for k in ("cx", "gates", "depth", "cx_depth")},
        fidelity=info["fidelity"], candidates=candidates,
    )
