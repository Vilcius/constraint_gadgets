"""Atomic local task files; checkpoints contain trusted, locally written pickles."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_value,
                                     allow_nan=False).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, default=json_value,
                                   allow_nan=False) + '\n').encode())


def write_pickle(path, value):
    atomic_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def environment():
    names = ['pennylane', 'pennylane-lightning', 'jax', 'jaxlib', 'optax',
             'numpy', 'scipy', 'matplotlib', 'pandas', 'nbformat', 'nbclient']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=versions)


def source_identity():
    files = list((ROOT / 'core').glob('*.py')) + [
        ROOT / 'tuning' / name for name in
        ['datasets.py', 'studies.py', 'storage.py', 'run.py']]
    files += [ROOT / 'run/generate_experiment_params.py',
              ROOT / 'data/make_data.py', ROOT / 'data/make_constraints.py']
    files += list((ROOT / 'data').glob('*.csv'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                     text=True).strip()
    return dict(commit=commit, files=hashes)


def ensure_run(directory, config):
    directory = Path(directory)
    provenance = dict(config=config, source=source_identity(), environment=environment())
    identifier = digest(provenance)
    path = directory / 'run.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['identifier'] != identifier:
            raise ValueError('Run configuration, source, or environment changed. '
                             'Use a new --output directory; existing results are unchanged.')
        return old
    result = dict(identifier=identifier, **provenance)
    write_json(path, result)
    return result


class TaskStore:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.path = self.directory / 'checkpoint.pkl'

    def load(self):
        import warnings
        found = False
        for path in [self.path, self.path.with_suffix('.previous.pkl')]:
            if not path.exists():
                continue
            found = True
            try:
                envelope = pickle.loads(path.read_bytes())
                payload = envelope['payload']
                if hashlib.sha256(payload).hexdigest() != envelope['sha256']:
                    raise ValueError('Checkpoint checksum mismatch')
                data = pickle.loads(payload)
            except (EOFError, pickle.UnpicklingError, ValueError, KeyError, TypeError):
                warnings.warn(f'Invalid checkpoint {path}; trying previous checkpoint')
                continue
            if data['identity'] != self.identity:
                raise ValueError('Checkpoint does not match this task/configuration')
            return data['state']
        if found:
            raise ValueError(f'No valid checkpoint in {self.directory}')
        return None

    def save(self, state):
        import jax
        import numpy as np
        state = jax.tree.map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x,
                             state)
        payload = pickle.dumps(dict(identity=self.identity, state=state),
                               protocol=pickle.HIGHEST_PROTOCOL)
        data = pickle.dumps(dict(payload=payload,
                                 sha256=hashlib.sha256(payload).hexdigest()))
        if self.path.exists():
            # Preserve only a valid predecessor, not a corrupt current file.
            try:
                envelope = pickle.loads(self.path.read_bytes())
                if hashlib.sha256(envelope['payload']).hexdigest() == envelope['sha256']:
                    atomic_bytes(self.path.with_suffix('.previous.pkl'), self.path.read_bytes())
            except (EOFError, pickle.UnpicklingError, ValueError, KeyError, TypeError):
                pass
        atomic_bytes(self.path, data)

    def status(self, status, **details):
        import datetime
        write_json(self.directory / 'status.json', dict(
            status=status, updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            identity=self.identity, **details))
