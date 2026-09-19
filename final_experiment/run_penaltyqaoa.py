#!/usr/bin/env python
"""Run the final PenaltyQAOA experiment independently."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_experiment.runner import main

if __name__ == "__main__":
    raise SystemExit(main("penaltyqaoa"))
