import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tuning'))
os.environ['JAX_ENABLE_X64'] = 'true'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MPLCONFIGDIR'] = str(ROOT/'tuning/.cache/matplotlib')
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
