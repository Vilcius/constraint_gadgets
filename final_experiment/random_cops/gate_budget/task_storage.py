"""Atomic NumPy checkpoint storage for gate-budget tasks."""
import hashlib
from pathlib import Path
import pickle
from tuning.storage import atomic_bytes, write_json


class TaskStore:
    """Checksum-protected pickle checkpoint with one valid predecessor."""

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.path = self.directory / "checkpoint.pkl"

    def load(self):
        found = False
        for path in (self.path, self.path.with_suffix(".previous.pkl")):
            if not path.exists():
                continue
            found = True
            try:
                envelope = pickle.loads(path.read_bytes())
                payload = envelope["payload"]
                if hashlib.sha256(payload).hexdigest() != envelope["sha256"]:
                    raise RuntimeError("checkpoint checksum mismatch")
                data = pickle.loads(payload)
                if data["identity"] != self.identity:
                    raise ValueError("Checkpoint identity mismatch")
                return data["state"]
            except (EOFError, pickle.UnpicklingError, KeyError, TypeError, RuntimeError):
                continue
        if found:
            raise ValueError(f"No valid checkpoint in {self.directory}")
        return None

    def save(self, state):
        payload = pickle.dumps(
            {"identity": self.identity, "state": state},
            protocol=pickle.HIGHEST_PROTOCOL)
        envelope = pickle.dumps({
            "payload": payload,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }, protocol=pickle.HIGHEST_PROTOCOL)
        previous = self.path.with_suffix(".previous.pkl")
        if self.path.exists():
            try:
                current = pickle.loads(self.path.read_bytes())
                if hashlib.sha256(current["payload"]).hexdigest() == current["sha256"]:
                    atomic_bytes(previous, self.path.read_bytes())
            except (EOFError, pickle.UnpicklingError, KeyError, TypeError):
                pass
        atomic_bytes(self.path, envelope)

    def status(self, status, **details):
        from datetime import datetime, timezone
        write_json(self.directory / "status.json", {
            "status": status,
            "updated": datetime.now(timezone.utc).isoformat(),
            "identity": self.identity,
            **details,
        })
