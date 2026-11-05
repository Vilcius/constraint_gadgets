#!/usr/bin/env python3
"""Build or resume the circuit-resource caches without running optimization."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from final_experiment.random_cops.resource_analysis import (
    estimate_final_resources, estimate_stateprep_resources)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stateprep-only", action="store_true")
    args = parser.parse_args()
    if args.stateprep_only:
        frame = estimate_stateprep_resources(
            ROOT / "final_experiment/random_cops/artifacts/manifest.json")
        print(frame.groupby(["method", "gate_set", "depth"]).size())
        return 0
    frame = estimate_final_resources(
        ROOT / "final_experiment/random_cops/artifacts/manifest.json",
        ROOT / "tuning/results/full/selected_vcg_db.pkl",
        cache_dir=ROOT / "final_experiment/random_cops/results/resource_estimates",
        depths=(1, 5),
    )
    print(frame.groupby(["method", "gate_set", "depth"]).size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
