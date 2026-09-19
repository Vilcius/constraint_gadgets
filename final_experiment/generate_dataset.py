#!/usr/bin/env python
"""Generate or audit the fixed 500-COP final experiment manifest."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "final_experiment/.cache/matplotlib"))

from final_experiment.datasets import audit, ensure_manifest, load_config
from final_experiment.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "final_experiment/config.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "final_experiment/artifacts/manifest.json")
    parser.add_argument("--check", action="store_true", help="audit the existing manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check:
        if not args.manifest.exists():
            parser.error(f"manifest does not exist: {args.manifest}")
        manifest = json.loads(args.manifest.read_text())
        if manifest["hash"] != __import__("final_experiment.storage", fromlist=["digest"]).digest(manifest["instances"]):
            raise ValueError("Dataset manifest hash mismatch")
        summary = audit(manifest["instances"], config)
        manifest["summary"] = summary
        write_json(args.manifest, manifest)
        write_json(args.manifest.with_name("dataset_summary.json"), summary)
    else:
        manifest = ensure_manifest(config, args.manifest)
        summary = manifest["summary"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
