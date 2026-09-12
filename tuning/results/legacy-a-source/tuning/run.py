#!/usr/bin/env python
"""Run from the repository root: python tuning/run.py --help."""
import argparse
from contextlib import contextmanager
import copy
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
sys.path.insert(0, str(ROOT/'tuning'))
from storage import ensure_run, write_json

STOP = None
SLOTS = None
SLOT = None


def runtime_environment(threads):
    cache = ROOT/'tuning/.cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache/'matplotlib'))
    os.environ.setdefault('JUPYTER_PLATFORM_DIRS', '1')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(cache/'jupyter'))
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = str(threads)
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['JAX_PLATFORMS'] = 'cpu'


def physical_cpus():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    seen, result = set(), []
    for cpu in allowed:
        path = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            key = ((path/'physical_package_id').read_text(), (path/'core_id').read_text())
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
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, SLOT)
    runtime_environment(threads)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())


def worker(job):
    try:
        if STOP.is_set():
            return dict(status='interrupted')
        from studies import run_a_instance, run_b_instance
        from core.optimization import TrainingInterrupted
        study, instance, config, output, run_id, manifest_hash = job
        try:
            (run_a_instance if study == 'a' else run_b_instance)(
                instance, config, output, run_id, manifest_hash, STOP.is_set)
            return dict(status='complete', instance=instance['id'])
        except TrainingInterrupted:
            return dict(status='interrupted', instance=instance['id'])
        except Exception:
            return dict(status='failed', instance=instance['id'], traceback=traceback.format_exc())
    finally:
        SLOTS.put(SLOT)


def configuration(path, profile):
    config = json.loads(Path(path).read_text())
    if profile == 'smoke':
        config['study_a'].update(supports=[4], per_support=4, qaoa_restarts=2,
                                qaoa_steps=2, ma_restarts=2, ma_steps=2, max_depth=2)
        config['study_b'].update(sizes=[4], per_regime=1, max_depth=2, restarts=2, steps=2)
        config['checkpoint_steps'] = 1
    config['profile'] = profile
    return config


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is generating/running in this output directory') from exc
        yield


def run_jobs(manifests, config, output, run_id, workers, threads):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from studies import settings_b
    from storage import digest
    context = mp.get_context('spawn')
    stop, slots = context.Event(), context.Queue()
    cpus = physical_cpus()
    workers = min(workers, len(cpus))
    per_worker = min(threads, max(1, len(cpus)//workers))
    for index in range(workers):
        slots.put(cpus[index*per_worker:(index+1)*per_worker])
    jobs = []
    for study, manifest in manifests.items():
        for instance in manifest['instances']:
            conditions = (config['study_a']['losses'] if study == 'a'
                          else settings_b(instance, config))
            complete = True
            for condition in conditions:
                label = condition if study == 'a' else condition['id']
                path = output/study/instance['id']/label/'result.json'
                identity = digest([run_id, manifest['hash'], instance['id'], condition])
                if not path.exists():
                    complete = False
                elif json.loads(path.read_text())['identity'] != identity:
                    raise ValueError(f'Completed result identity mismatch: {path}')
            if not complete:
                jobs.append((study, instance, config, str(output), run_id, manifest['hash']))
    if not jobs:
        print('All requested tasks are complete; nothing to resume.')
        return 0
    print(f'{len(jobs)} instance groups; {workers} workers × {per_worker} physical CPU cores', flush=True)
    old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    failures = 0
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
             initializer=initialize_worker, initargs=(slots, stop, per_worker),
             max_tasks_per_child=1) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'failed':
                    failures += 1
                    print(result['traceback'], file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        slots.close()
    return 130 if stop.is_set() else (1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'run', 'resume', 'status', 'export', 'smoke', 'preflight'])
    parser.add_argument('--study', choices=['a', 'b', 'both'], default='both')
    parser.add_argument('--profile', choices=['full', 'smoke'], default='full')
    parser.add_argument('--config', type=Path, default=ROOT/'tuning/config.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error('workers and threads must be positive')
    if args.command in ['smoke', 'preflight']:
        args.profile = 'smoke'
    runtime_environment(args.threads)
    config = configuration(args.config, args.profile)
    output = (args.output or ROOT/'tuning/results'/args.profile).resolve()
    if args.command in ['status', 'export']:
        from analyze import status_summary, export_results
        if args.command == 'status':
            print(json.dumps(status_summary(output), indent=2))
        else:
            export_results(output)
        return 0
    from datasets import ensure_manifest
    with output_lock(output):
        provenance = ensure_run(output, config)
        studies = ['a', 'b'] if args.study == 'both' else [args.study]
        manifests = {study: ensure_manifest(output, config, study) for study in studies}
        print(f'Output: {output}', flush=True)
        for study, manifest in manifests.items():
            print(f'Study {study.upper()}: {len(manifest["instances"])} fixed instances', flush=True)
        if args.command == 'generate':
            return 0
        result = run_jobs(manifests, config, output, provenance['identifier'], args.workers, args.threads)
        from analyze import export_results, status_summary
        export_results(output)
        write_json(output/'execution_summary.json', status_summary(output))
        return result


if __name__ == '__main__':
    raise SystemExit(main())
