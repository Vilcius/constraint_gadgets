"""Status and flat exports for independent final-experiment method runs."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd


def status_summary(output, expected=None):
    output = Path(output)
    if expected is None:
        selection = output / "selection.json"
        if selection.exists():
            expected = len(json.loads(selection.read_text())["instance_ids"])
        elif (output / "run.json").exists():
            run = json.loads((output / "run.json").read_text())
            expected = 2 if run["config"].get("profile") == "smoke" else 500
        else:
            expected = 500
    counts = Counter()
    for path in output.glob("final_*/*status.json"):
        counts[json.loads(path.read_text())["status"]] += 1
    completed = len(list(output.glob("final_*/result.json")))
    counts["pending"] = max(0, expected - sum(counts.values()))
    return {"expected": expected, "completed_results": completed,
            "statuses": dict(counts)}


def export_results(output):
    output = Path(output)
    summaries = [json.loads(path.read_text()) for path in sorted(output.glob("final_*/result.json"))]
    depths = []
    for path in sorted(output.glob("final_*/depths.json")):
        instance_id = path.parent.name
        for row in json.loads(path.read_text()):
            depths.append({"instance_id": instance_id,
                           **{key: value for key, value in row.items()
                              if key not in ["angles", "history", "restarts",
                                             "counts", "probabilities"]}})
    pd.DataFrame(summaries).to_csv(output / "results.csv", index=False)
    pd.DataFrame(depths).to_csv(output / "depths.csv", index=False)
