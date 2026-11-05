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
from core.qaoa_plus import QAOAPlus
from core.pc_qaoa import PCQAOA
from core import constraint_handler as ch


ROOT = Path(__file__).resolve().parents[2]


def estimate_stateprep_resources(manifest_path, cache_dir=None, depths=(1, 5)):
    """Count the executed StatePrep variant, with an independent resumable cache."""
    manifest_path = Path(manifest_path)
    cache_dir = Path(cache_dir or ROOT /
                     "final_experiment/random_cops/results/stateprep_resource_estimates")
    cache_dir.mkdir(parents=True, exist_ok=True)
    shared = {
        "schema": 1, "depths": list(map(int, depths)),
        "gate_sets": {k: sorted(v) for k, v in re.GATE_SETS.items()},
        "sources": {name: _sha256(ROOT / name) for name in (
            "core/constraint_state_prep.py", "core/pc_qaoa.py",
            "core/qaoa_base.py", "core/constraint_handler.py",
            "core/dicke_state_prep.py", "core/resource_estimation.py",
            "final_experiment/random_cops/resource_analysis.py")},
    }
    rows = []
    instances = json.loads(manifest_path.read_text())["instances"]
    for index, instance in enumerate(instances, 1):
        identity = hashlib.sha256(json.dumps(
            {"shared": shared, "instance": instance}, sort_keys=True).encode()).hexdigest()
        path = cache_dir / f"{instance['id']}.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        if saved.get("identity") == identity:
            rows.extend(saved["rows"])
            continue
        solver = PCQAOA(
            np.asarray(instance["Q"], dtype=float),
            ch.parse_constraints(instance["constraints"]),
            instance["structural_indices"], instance["penalty_indices"],
            penalty_pen=float(instance["delta"]), angle_strategy="QAOA",
            n_layers=1, gadget_preparation="stateprep", compile_on_init=False,
        )
        resources = re.estimate_pc_qaoa_resources(solver)
        instance_rows = []
        for gate_set, item in resources.items():
            for depth in depths:
                full = item["sp"].add_series(item["layer"].multiply_series(int(depth)))
                instance_rows.append(_row(
                    instance, "PC-QAOA (StatePrep)", gate_set, depth, full,
                    solver.n_total, solver.n_slack))
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"identity": identity, "rows": instance_rows},
                                        indent=2) + "\n")
        temporary.replace(path)
        rows.extend(instance_rows)
        if index % 25 == 0 or index == len(instances):
            print(f"StatePrep resources: {index}/{len(instances)}", flush=True)
    frame = pd.DataFrame(rows)
    csv = cache_dir / "stateprep_circuit_resources.csv"
    temporary = csv.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv)
    return frame


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
    cache_dir = Path(cache_dir or ROOT / "final_experiment/random_cops/results/resource_estimates")
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "final_circuit_resources.csv"
    metadata_path = cache_dir / "final_circuit_resources.json"

    shared_identity = {
        "schema": 2,
        "vcg_database": _sha256(vcg_database_path),
        "sources": {
            name: _sha256(ROOT / name) for name in [
                "final_experiment/random_cops/resource_analysis.py",
                "core/resource_estimation.py",
                "core/qaoa_base.py",
                "core/pc_qaoa.py",
                "core/penalty_qaoa.py",
                "core/qaoa_plus.py",
                "core/dicke_state_prep.py",
                "core/vcg.py",
            ]
        },
        "depths": [int(depth) for depth in depths],
        "gate_sets": {name: sorted(gates) for name, gates in re.GATE_SETS.items()},
    }
    identity = {**shared_identity, "manifest": _sha256(manifest_path)}
    if csv_path.exists() and metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved.get("identity") == identity:
            return pd.read_csv(csv_path)

    manifest = json.loads(manifest_path.read_text())
    vcg_database = pickle.loads(vcg_database_path.read_bytes())
    rows = []
    max_depth = max(int(depth) for depth in depths)
    instance_cache = cache_dir / "instances"
    instance_cache.mkdir(parents=True, exist_ok=True)
    for index, instance in enumerate(manifest["instances"], 1):
        item_identity = hashlib.sha256(json.dumps(
            {"shared": shared_identity, "instance": instance},
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        item_path = instance_cache / f"{instance['id']}.json"
        if item_path.exists():
            saved_item = json.loads(item_path.read_text())
            if saved_item.get("identity") == item_identity:
                rows.extend(saved_item["rows"])
                continue
        instance_rows = []
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
        qaoa_plus_model = QAOAPlus(
            np.asarray(instance["Q"], dtype=float), instance["constraints"],
            angle_strategy="QAOA", n_layers=1,
            feasible_states=instance["feasible_bitstrings"], compile_on_init=False,
        )
        qaoa_plus_resources = re.estimate_qaoa_plus_resources(qaoa_plus_model)
        for gate_set in re.GATE_SETS:
            method_resources = [
                ("PC-QAOA", estimate[gate_set]["pc_qaoa_sp"],
                 estimate[gate_set]["pc_qaoa_layer"],
                 estimate["n_qubits_pc"], estimate["n_slack_pc"]),
                ("PenaltyQAOA", executed_penalty[gate_set]["sp"],
                 executed_penalty[gate_set]["layer"],
                 penalty_model.n_total, penalty_model.n_slack),
                ("QAOA+", qaoa_plus_resources[gate_set]["sp"],
                 qaoa_plus_resources[gate_set]["layer"],
                 qaoa_plus_model.n_total, 0),
            ]
            for method, preparation, layer, logical_qubits, slack_qubits in method_resources:
                for depth in depths:
                    full = preparation.add_series(layer.multiply_series(int(depth)))
                    instance_rows.append(_row(instance, method, gate_set, depth, full,
                                              logical_qubits, slack_qubits))
        temporary_item = item_path.with_suffix(".json.tmp")
        temporary_item.write_text(json.dumps(
            {"identity": item_identity, "rows": instance_rows}, indent=2) + "\n")
        temporary_item.replace(item_path)
        rows.extend(instance_rows)
        if index % 25 == 0 or index == len(manifest["instances"]):
            print(f"Resource estimates: {index}/{len(manifest['instances'])}", flush=True)

    frame = pd.DataFrame(rows)
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps({"identity": identity,
                                               "rows": len(frame)}, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    return frame
