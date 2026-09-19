"""Resumable single-COP execution for the two final experiment methods."""
import copy
import json
import os
from pathlib import Path
import pickle
import resource
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "final_experiment/.cache/matplotlib"))

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from core import constraint_handler as ch
from core.optimization import (make_adam_functions, optimize_resumable,
                               TrainingInterrupted)
from core.pc_qaoa import PCQAOA
from core.penalty_qaoa import PenaltyQAOA
from final_experiment.storage import TaskStore, digest, write_json


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:8], 16)


def probability_metrics(probabilities, reference):
    probabilities = np.asarray(probabilities, dtype=float)
    mask = np.asarray(reference["feasible_mask"], dtype=bool)
    optimal = np.asarray(reference["optimal_indices"], dtype=int)
    p_feasible = float(probabilities[mask].sum())
    p_optimal = float(probabilities[optimal].sum())
    values = np.asarray(reference["objective_values"])
    conditional = float(probabilities[mask] @ values[mask] / p_feasible) if p_feasible else None
    spread = reference["f_max_feasible"] - reference["f_star"]
    quality = ((reference["f_max_feasible"] - conditional) / spread
               if spread and p_feasible else (1.0 if p_feasible else None))
    return {"p_feasible": p_feasible, "p_optimal": p_optimal,
            "conditional_objective": conditional, "conditional_quality": quality,
            "weighted_quality": p_feasible * quality if p_feasible else 0.0}


def optimizer_phase(task, store, phase, cost, step, *, config, shape, steps,
                    restarts, lr, seed, args, previous=None,
                    should_stop=lambda: False):
    active = task.get("optimizer")
    if active is not None and task.get("phase") != phase:
        raise ValueError("Checkpoint phase mismatch")
    task["phase"] = phase

    def checkpoint(state):
        task["optimizer"] = state
        store.save(task)
        store.status("running", phase=phase, restart=state["restart"], step=state["step"])

    return optimize_resumable(
        cost, step, shape=shape, steps=steps, restarts=restarts,
        learning_rate=lr, seed=seed, args=args, previous=previous,
        state=active, checkpoint=checkpoint,
        checkpoint_steps=config["checkpoint_steps"], should_stop=should_stop,
    )


def timing(result):
    return {"compile_seconds": float(result["compile_seconds"]),
            "optimize_seconds": float(result["optimize_seconds"])}


def _normalize(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Invalid final probability distribution")
    return probabilities / total


def _sample(probabilities, instance, config, depth):
    sample_seed = seed_for(config["sampling_seed"], instance["id"], depth)
    counts = np.random.default_rng(sample_seed).multinomial(config["shots"], probabilities)
    exact = probability_metrics(probabilities, instance)
    sampled = probability_metrics(counts / config["shots"], instance)
    return {
        "counts": counts.tolist(), "probabilities": probabilities.tolist(),
        "sampling_seed": sample_seed, "shots": config["shots"],
        **sampled, **{f"exact_{key}": value for key, value in exact.items()},
    }


def _pc_functions(instance, config, depth):
    parsed = ch.parse_constraints(instance["constraints"])
    db_path = Path(config["vcg_database"])
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    database = pickle.loads(db_path.read_bytes())
    key = ch.normalize_constraint(instance["vcg_constraint"])
    if key not in database:
        raise ValueError(f"VCG database does not contain {instance['vcg_id']}: {key}")
    solver = PCQAOA(
        qubo=np.asarray(instance["Q"]), all_constraints=parsed,
        structural_indices=instance["structural_indices"],
        penalty_indices=instance["penalty_indices"],
        penalty_pen=float(instance["delta"]), angle_strategy="QAOA",
        n_layers=depth, learning_rate=config["learning_rate"],
        steps=config["steps"], num_restarts=config["restarts"],
        gadget_db_path=str(db_path),
    )
    dev = qml.device("default.qubit", wires=solver.all_wires)

    @qml.qnode(dev, interface="jax", diff_method=None)
    def probabilities(angles):
        solver.circuit(angles)
        return qml.probs(wires=solver.x_wires)

    return solver, solver._compiled_cost, solver._compiled_step, jax.jit(probabilities)


def _penalty_template(instance, config):
    return PenaltyQAOA(
        np.asarray(instance["Q"]), instance["constraints"], penalty=1.0,
        angle_strategy="QAOA", learning_rate=config["learning_rate"],
        steps=config["steps"], num_restarts=config["restarts"],
        compile_on_init=False,
    )


def run_instance(method, instance, config, directory, run_id, manifest_hash,
                 should_stop=lambda: False):
    identity = digest([run_id, manifest_hash, method, instance["id"]])
    store = TaskStore(Path(directory) / instance["id"], identity)
    result_path = store.directory / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing["identity"] != identity:
            raise ValueError(f"Completed result identity mismatch: {result_path}")
        return
    task = store.load() or {"depths": [], "optimizer": None}
    penalty_template = _penalty_template(instance, config) if method == "penaltyqaoa" else None
    try:
        for depth in range(len(task["depths"]) + 1, config["max_depth"] + 1):
            if method == "pcqaoa":
                solver, cost, step, probabilities_fn = _pc_functions(instance, config, depth)
                n_slack, n_total = solver.n_slack, solver.n_total
            elif method == "penaltyqaoa":
                solver = copy.copy(penalty_template)
                solver.n_layers = depth
                objective, probabilities_fn = solver.tuning_functions()
                cost, step = make_adam_functions(objective, config["learning_rate"])
                n_slack, n_total = solver.n_slack, solver.n_total
            else:
                raise ValueError(method)
            previous = task["depths"][-1]["angles"] if task["depths"] else None
            args = (() if method == "pcqaoa" else (jnp.asarray(instance["delta"]),))
            optimized = optimizer_phase(
                task, store, f"depth_{depth}", cost, step,
                config=config, shape=(depth, 2), steps=config["steps"],
                restarts=config["restarts"], lr=config["learning_rate"],
                seed=seed_for(config["training_seed"], instance["id"], depth),
                args=args, previous=previous, should_stop=should_stop,
            )
            begin = time.perf_counter()
            probabilities = _normalize(probabilities_fn(optimized["best_angles"], *args))
            jax.block_until_ready(probabilities)
            probability_seconds = time.perf_counter() - begin
            row = {
                "depth": depth, "angles": optimized["best_angles"],
                "loss": optimized["best_loss"], "history": optimized["history"],
                "restarts": optimized["restart_results"], "n_slack": n_slack,
                "n_total": n_total, "probability_seconds": probability_seconds,
                **_sample(probabilities, instance, config, depth), **timing(optimized),
            }
            task["depths"].append(row)
            task["optimizer"] = None
            store.save(task)
            write_json(store.directory / "depths.json", task["depths"])
        final = task["depths"][-1]
        summary = {
            "identity": identity, "instance_id": instance["id"], "method": method,
            "vcg_id": instance["vcg_id"], "vcg_family": instance["vcg_family"],
            "vcg_support": instance["support"], "n_x": instance["n_x"],
            "m": instance["m"], "o": instance["o"], "regime": instance["regime"],
            "n_structural": instance["n_structural"], "n_penalty": instance["n_penalty"],
            "delta": instance["delta"], "depth": final["depth"],
            "n_slack": final["n_slack"], "n_total": final["n_total"],
            "p_optimal": final["p_optimal"], "p_feasible": final["p_feasible"],
            "exact_p_optimal": final["exact_p_optimal"],
            "exact_p_feasible": final["exact_p_feasible"],
            "compile_seconds": sum(row["compile_seconds"] for row in task["depths"]),
            "optimize_seconds": sum(row["optimize_seconds"] for row in task["depths"]),
            "probability_seconds": sum(row["probability_seconds"] for row in task["depths"]),
            "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        write_json(result_path, summary)
        store.status("complete")
        print(f"{method} {instance['id']}: P(opt)={final['p_optimal']:.4f} at p={final['depth']}",
              flush=True)
    except TrainingInterrupted:
        store.status("interrupted", phase=task.get("phase"))
        raise
    except Exception:
        store.status("failed", phase=task.get("phase"), traceback=traceback.format_exc())
        raise
