"""Full-circuit resource estimates for the fixed final experiment."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml

from core import resource_estimation as re
from core import qaoa_base as base
from core.penalty_qaoa import PenaltyQAOA


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(resources) -> dict[str, int]:
    return dict(resources.gate_counts)


def _row(instance, method, gate_set, depth, resources, logical_qubits, slack_qubits):
    counts = _counts(resources)
    return {
        "instance_id": instance["id"],
        "method": method,
        "gate_set": gate_set,
        "depth": int(depth),
        "n_x": int(instance["n_x"]),
        "support": int(instance["support"]),
        "regime": instance["regime"],
        "overlap_level": int(instance["o"]),
        "m": int(instance["m"]),
        "logical_qubits": int(logical_qubits),
        "slack_qubits": int(slack_qubits),
        "total_wires": int(resources.algo_wires + resources.zeroed_wires
                           + resources.any_state_wires),
        "algorithmic_wires": int(resources.algo_wires),
        "allocated_wires": int(resources.zeroed_wires + resources.any_state_wires),
        "total_gates": int(sum(counts.values())),
        "one_qubit_gates": int(sum(value for gate, value in counts.items()
                                    if gate in re._1Q_GATES_SET)),
        "two_qubit_gates": int(sum(value for gate, value in counts.items()
                                    if gate in re._2Q_GATES_SET)),
        "t_gates": int(counts.get("T", 0)),
    }


def _executed_penalty_resources(instance):
    """Match the coalesced Hamiltonian traced by PenaltyQAOA.tuning_functions."""
    model = PenaltyQAOA(
        np.asarray(instance["Q"], dtype=float), instance["constraints"],
        penalty=1.0, angle_strategy="QAOA", n_layers=1, compile_on_init=False,
    )
    unit_penalty = base.build_penalty_hamiltonian(
        model.parsed_constraints, model.slack_info, 1.0,
        fallback_wire=model.x_wires[0],
    )
    q_rep, p_rep = model.qubo_Ham.pauli_rep, unit_penalty.pauli_rep
    words = list(dict.fromkeys([*q_rep, *p_rep]))
    words = [word for word in words
             if q_rep.get(word, 0) != 0 or p_rep.get(word, 0) != 0]
    delta = float(instance["delta"])
    coefficients = [float(np.real(q_rep.get(word, 0) + delta * p_rep.get(word, 0)))
                    for word in words]
    hamiltonian = qml.Hamiltonian(
        coefficients, [word.operation(wire_order=model.all_wires) for word in words])
    return model, re.estimate_penalty_resources(model, circuit_hamiltonian=hamiltonian)


def estimate_final_resources(manifest_path, vcg_database_path, cache_dir=None,
                             depths=(1, 5)) -> pd.DataFrame:
    """Estimate the circuits defined by the saved manifest without simulation."""
    manifest_path = Path(manifest_path)
    vcg_database_path = Path(vcg_database_path)
    cache_dir = Path(cache_dir or ROOT / "final_experiment/results/resource_estimates")
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "final_circuit_resources.csv"
    metadata_path = cache_dir / "final_circuit_resources.json"

    identity = {
        "schema": 1,
        "manifest": _sha256(manifest_path),
        "vcg_database": _sha256(vcg_database_path),
        "sources": {
            name: _sha256(ROOT / name) for name in [
                "final_experiment/resource_analysis.py",
                "core/resource_estimation.py",
                "core/qaoa_base.py",
                "core/pc_qaoa.py",
                "core/penalty_qaoa.py",
                "core/dicke_state_prep.py",
                "core/vcg.py",
            ]
        },
        "depths": [int(depth) for depth in depths],
        "gate_sets": {name: sorted(gates) for name, gates in re.GATE_SETS.items()},
    }
    if csv_path.exists() and metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved.get("identity") == identity:
            return pd.read_csv(csv_path)

    manifest = json.loads(manifest_path.read_text())
    vcg_database = pickle.loads(vcg_database_path.read_bytes())
    rows = []
    max_depth = max(int(depth) for depth in depths)
    for instance in manifest["instances"]:
        task = {"constraints": instance["constraints"], "n_x": instance["n_x"],
                "qubo_idx": 0}
        qubos = {instance["n_x"]: [{"Q": np.asarray(instance["Q"], dtype=float)}]}
        estimate = re.estimate_from_task(
            task, qubos, n_layers=max_depth, penalty_weight=float(instance["delta"]),
            vcg_db=vcg_database,
            structural_indices=instance["structural_indices"],
            penalty_indices=instance["penalty_indices"],
        )
        if estimate["vcg_missing_pc"]:
            raise ValueError(f"Missing saved VCG for {instance['id']}")
        penalty_model, executed_penalty = _executed_penalty_resources(instance)
        for gate_set in re.GATE_SETS:
            method_resources = [
                ("PC-QAOA", estimate[gate_set]["pc_qaoa_sp"],
                 estimate[gate_set]["pc_qaoa_layer"],
                 estimate["n_qubits_pc"], estimate["n_slack_pc"]),
                ("PenaltyQAOA", executed_penalty[gate_set]["sp"],
                 executed_penalty[gate_set]["layer"],
                 penalty_model.n_total, penalty_model.n_slack),
            ]
            for method, preparation, layer, logical_qubits, slack_qubits in method_resources:
                for depth in depths:
                    full = preparation.add_series(layer.multiply_series(int(depth)))
                    rows.append(_row(instance, method, gate_set, depth, full,
                                     logical_qubits, slack_qubits))

    frame = pd.DataFrame(rows)
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps({"identity": identity,
                                               "rows": len(frame)}, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    return frame
