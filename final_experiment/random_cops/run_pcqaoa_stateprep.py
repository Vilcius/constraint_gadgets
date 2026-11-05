#!/usr/bin/env python
"""Run PC-QAOA with exact StatePrep replacing the trained VCG circuits."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_experiment.random_cops.runner import main

if __name__ == "__main__":
    raise SystemExit(main("pcqaoa_stateprep"))
