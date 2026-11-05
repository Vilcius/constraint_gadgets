#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from final_experiment.random_cops.gate_budget.runner import main
if __name__=='__main__':raise SystemExit(main('pcqaoa'))
