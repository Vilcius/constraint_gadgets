"""Provenance and storage helpers for the final paired experiment."""
import hashlib
import json
from pathlib import Path
import subprocess

from tuning.storage import (TaskStore, atomic_bytes, digest, environment,
                            write_json, write_pickle)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "final_experiment"


def source_identity():
    files = list((ROOT / "core").glob("*.py"))
    files += [
        HERE / "config.json",
        HERE / "datasets.py",
        HERE / "runner.py",
        HERE / "storage.py",
        HERE / "studies.py",
    ]
    files += [
        ROOT / "tuning" / "storage.py",
        ROOT / "tuning" / "artifacts" / "selected_vcg_db.pkl",
        ROOT / "tuning" / "artifacts" / "manifest_a.json",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files if path.exists()
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {"commit": commit, "files": hashes}


def ensure_run(directory, config, method, manifest_hash):
    directory = Path(directory)
    provenance = {
        "method": method,
        "manifest_hash": manifest_hash,
        "config": config,
        "source": source_identity(),
        "environment": environment(),
    }
    identifier = digest(provenance)
    path = directory / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["identifier"] != identifier:
            raise ValueError(
                "Configuration, dataset, source, or environment changed. "
                "Use a new output directory."
            )
        return old
    result = {"identifier": identifier, **provenance}
    write_json(path, result)
    return result
