"""Shared bounded-process runner used by the two method-specific scripts."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "final_experiment/.cache/matplotlib"))

from final_experiment.analyze import export_results, status_summary
from final_experiment.datasets import ensure_manifest, load_config
from final_experiment.storage import ensure_run, write_json

STOP = None
SLOTS = None
SLOT = None


def runtime_environment(threads):
    cache = ROOT / "final_experiment/.cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(cache / "jupyter"))
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = str(threads)
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"


def physical_cpus():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    seen, result = set(), []
    for cpu in allowed:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((path / "physical_package_id").read_text(), (path / "core_id").read_text())
        except OSError:
            key = cpu
        if key not in seen:
            seen.add(key)
            result.append(cpu)
    return result


def initialize_worker(slots, stop, threads):
    global STOP, SLOTS, SLOT
    STOP, SLOTS = stop, slots
    SLOT = slots.get()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, SLOT)
    runtime_environment(threads)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())


def worker(job):
    try:
        if STOP.is_set():
            return {"status": "interrupted"}
        from core.optimization import TrainingInterrupted
        from final_experiment.studies import run_instance
        method, instance, config, output, run_id, manifest_hash = job
        try:
            run_instance(method, instance, config, output, run_id, manifest_hash, STOP.is_set)
            return {"status": "complete", "instance": instance["id"]}
        except TrainingInterrupted:
            return {"status": "interrupted", "instance": instance["id"]}
        except Exception:
            return {"status": "failed", "instance": instance["id"],
                    "traceback": traceback.format_exc()}
    finally:
        SLOTS.put(SLOT)


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another command is writing to {output}") from error
        yield


def _profile_config(config, profile):
    config = dict(config)
    if profile == "smoke":
        config.update(max_depth=2, restarts=2, steps=2, shots=1000,
                      checkpoint_steps=1)
    config["profile"] = profile
    return config


def _smoke_instances(instances):
    ordered = sorted(instances, key=lambda row: (row["n_total_penaltyqaoa"], row["n_x"], row["id"]))
    disjoint = next(row for row in ordered if row["regime"] == "disjoint")
    overlapping = next(row for row in ordered if row["regime"] == "overlapping")
    return [disjoint, overlapping]


def main(method):
    parser = argparse.ArgumentParser(description=f"Run final {method} tasks")
    parser.add_argument("command", choices=["run", "resume", "smoke", "status", "export"])
    parser.add_argument("--config", type=Path, default=ROOT / "final_experiment/config.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "final_experiment/artifacts/manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")
    profile = "smoke" if args.command == "smoke" else "full"
    default = ROOT / "final_experiment/results" / (f"smoke_{method}" if profile == "smoke" else method)
    output = (args.output or default).resolve()
    if args.command == "status":
        print(json.dumps(status_summary(output), indent=2))
        return 0
    if args.command == "export":
        export_results(output)
        print(json.dumps(status_summary(output), indent=2))
        return 0
    runtime_environment(args.threads)
    config = _profile_config(load_config(args.config), profile)
    manifest = ensure_manifest(config, args.manifest)
    instances = _smoke_instances(manifest["instances"]) if profile == "smoke" else manifest["instances"]
    instances = [row for index, row in enumerate(instances)
                 if index % args.shard_count == args.shard_index]
    with output_lock(output):
        provenance = ensure_run(output, config, method, manifest["hash"])
        selection_path = output / "selection.json"
        selection = {"instance_ids": [row["id"] for row in instances],
                     "shard_count": args.shard_count, "shard_index": args.shard_index}
        if selection_path.exists():
            if json.loads(selection_path.read_text()) != selection:
                raise ValueError("Output directory was created for a different instance selection")
        else:
            write_json(selection_path, selection)
        jobs = []
        for instance in instances:
            result_path = output / instance["id"] / "result.json"
            if result_path.exists():
                existing = json.loads(result_path.read_text())
                from final_experiment.storage import digest
                expected = digest([provenance["identifier"], manifest["hash"], method, instance["id"]])
                if existing["identity"] != expected:
                    raise ValueError(f"Completed result identity mismatch: {result_path}")
                continue
            jobs.append((method, instance, config, str(output), provenance["identifier"], manifest["hash"]))
        if not jobs:
            print("All selected tasks are complete; nothing to resume.")
            return 0
        cpus = physical_cpus()
        workers = min(args.workers, len(cpus), len(jobs))
        per_worker = min(args.threads, max(1, len(cpus) // workers))
        print(f"{len(jobs)} {method} COPs; {workers} workers × {per_worker} physical CPU cores",
              flush=True)
        context = mp.get_context("spawn")
        stop, slots = context.Event(), context.Queue()
        for index in range(workers):
            slots.put(cpus[index * per_worker:(index + 1) * per_worker])
        old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        failures = 0
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context, initializer=initialize_worker,
                initargs=(slots, stop, per_worker), max_tasks_per_child=1,
            ) as pool:
                futures = [pool.submit(worker, job) for job in jobs]
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "failed":
                        failures += 1
                        print(result["traceback"], file=sys.stderr, flush=True)
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            slots.close()
        export_results(output)
        write_json(output / "execution_summary.json",
                   status_summary(output, expected=len(instances)))
        return 130 if stop.is_set() else (1 if failures else 0)
