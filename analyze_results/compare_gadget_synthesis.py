"""
compare_gadget_synthesis.py -- Circuit size of trained ma-QAOA VCGs vs.
exact state-preparation synthesis, per constraint.

For every trained gadget in each VCG database:

  VCG, three ways of counting the same trained circuit
    vcg_ladder      MultiRZ(k) -> 2(k-1) CNOT ladder, no optimisation
                    (the convention used for the paper's resource figures)
    vcg_transpiled  the ladder circuit through qiskit optimization_level=3
    vcg_diagonal    each cost layer recompiled as one diagonal unitary
                    (Gray-code synthesis), then level 3 -- the cheapest
                    honest compilation of the ma-QAOA ansatz we know of

  Synthesis, every back-end in core.synth_gadget.SYNTHESISERS, each
  transpiled to {cx, u} at level 3 (best of 5 seeds).

Fidelity to the uniform feasible state is reported for every circuit.

Usage
-----
    python analyze_results/compare_gadget_synthesis.py
    python analyze_results/compare_gadget_synthesis.py --db gadgets/vcg_db.pkl:main
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate

from core import synth_gadget as sg
from core.vcg import VCG

DEFAULT_DBS = [
    "gadgets/vcg_db.pkl:paper_v2",
    "tuning/results/full/selected_vcg_db.pkl:study_a",
]


def restore_vcg(key: str, entry: dict) -> VCG:
    """Rebuild a trained VCG from a database entry without retraining."""
    gadget = VCG(constraints=[key])
    gadget.opt_angles = np.asarray(entry["opt_angles"])
    gadget.n_layers = int(entry["n_layers"])
    return gadget


def vcg_to_qiskit(gadget: VCG) -> tuple[QuantumCircuit, list]:
    """Translate the trained VCG tape; also return its cost-layer diagonals."""
    n = gadget.n_x
    q = {w: n - 1 - i for i, w in enumerate(gadget.var_wires)}
    tape = qml.tape.make_qscript(gadget.opt_circuit)()
    qc = QuantumCircuit(n)
    layers, phases = [], None   # per-layer list of diagonal phase vectors
    z = np.array(list(np.ndindex(*(2,) * n)))          # rows = bitstrings, MSB first
    pos = {w: i for i, w in enumerate(gadget.var_wires)}
    for op in tape.operations:
        name = op.name
        if name == "Hadamard":
            qc.h(q[op.wires[0]])
        elif name == "RX":
            if phases is not None:
                layers.append(phases)
                phases = None
            qc.rx(float(op.parameters[0]), q[op.wires[0]])
        elif name == "MultiRZ":
            theta = float(op.parameters[0])
            wires = [q[w] for w in op.wires]
            for a, b in zip(wires, wires[1:]):
                qc.cx(a, b)
            qc.rz(theta, wires[-1])
            for a, b in reversed(list(zip(wires, wires[1:]))):
                qc.cx(a, b)
            parity = z[:, [pos[w] for w in op.wires]].sum(axis=1) % 2
            term = -theta / 2 * (1 - 2 * parity)        # exp(-i theta/2 Z...Z)
            phases = term if phases is None else phases + term
        else:
            raise ValueError(f"Unexpected op {name} in VCG tape")
    if phases is not None:
        layers.append(phases)
    return qc, layers


def vcg_diagonal_circuit(gadget: VCG, layers: list) -> QuantumCircuit:
    """Same ansatz with each cost layer compiled as a single diagonal."""
    n = gadget.n_x
    tape = qml.tape.make_qscript(gadget.opt_circuit)()
    rx = [op for op in tape.operations if op.name == "RX"]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    q = {w: n - 1 - i for i, w in enumerate(gadget.var_wires)}
    for layer, phases in enumerate(layers):
        qc.append(DiagonalGate(list(np.exp(1j * phases))), range(n))
        for op in rx[layer * n:(layer + 1) * n]:
            qc.rx(float(op.parameters[0]), q[op.wires[0]])
    return qc


def compare_entry(key: str, entry: dict, db_name: str, seeds) -> list[dict]:
    wires, mask = sg.feasible_mask(key)
    vec = sg.target_state(mask)
    base = {"db": db_name, "constraint": key, "support": len(wires),
            "n_feasible": int(mask.sum()), "vcg_layers": int(entry["n_layers"])}
    rows = []

    gadget = restore_vcg(key, entry)
    ladder, layers = vcg_to_qiskit(gadget)
    circuits = {
        "vcg_ladder": ladder,
        "vcg_transpiled": sg.compile_min_cx(ladder, seeds),
        "vcg_diagonal": sg.compile_min_cx(vcg_diagonal_circuit(gadget, layers), seeds),
    }
    for method, qc in circuits.items():
        rows.append({**base, "method": method, **sg.circuit_metrics(qc),
                     "fidelity": sg.fidelity(qc, vec)})

    gadget_synth = sg.synthesize(key, seeds=seeds, fidelity_tol=1.0)
    for method, info in gadget_synth.candidates.items():
        rows.append({**base, "method": method, **info})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", action="append",
                        help="path:label of a VCG database (repeatable)")
    parser.add_argument("--output", default="results/gadget_synthesis")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    seeds = range(args.seeds)
    rows = []
    for spec in args.db or DEFAULT_DBS:
        path, label = spec.rsplit(":", 1)
        with open(path, "rb") as f:
            db = pickle.load(f)
        trained = {k: e for k, e in db.items() if e.get("opt_angles") is not None}
        print(f"{label}: {len(trained)} trained gadgets of {len(db)} entries")
        for i, (key, entry) in enumerate(sorted(trained.items())):
            start = time.time()
            rows.extend(compare_entry(key, entry, label, seeds))
            print(f"  [{i + 1}/{len(trained)}] {key}  ({time.time() - start:.1f}s)",
                  flush=True)

    os.makedirs(args.output, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output, "comparison.csv"), index=False)

    ok = df[df["error"].isna()] if "error" in df else df
    exact = ok[ok["method"].str.startswith(("qiskit", "qclib"))]
    exact = exact[exact["fidelity"] > 1 - 1e-8]
    best = (exact.sort_values(["cx", "depth"])
            .groupby(["db", "constraint"], as_index=False).first()
            .assign(method="synth_best"))
    table = pd.concat([ok, best], ignore_index=True)
    summary = (table.groupby(["db", "support", "method"])
               .agg(n=("constraint", "nunique"), cx_mean=("cx", "mean"),
                    cx_max=("cx", "max"), depth_mean=("depth", "mean"),
                    fid_min=("fidelity", "min"))
               .round(3))
    summary.to_csv(os.path.join(args.output, "summary.csv"))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary)


if __name__ == "__main__":
    main()
