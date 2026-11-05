"""CPU affinity, thread limits, and output locks for gate-budget runs."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal

HERE = Path(__file__).resolve().parents[1]
STOP = SLOTS = SLOT = None


def runtime_environment(threads):
    cache = HERE / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(threads)


def physical_cpus():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    seen, result = set(), []
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((topology / "physical_package_id").read_text(),
                   (topology / "core_id").read_text())
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


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another command is writing to {output}") from error
        yield
