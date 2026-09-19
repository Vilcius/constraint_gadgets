# Final paired PC-QAOA and PenaltyQAOA experiment

This folder is part of the existing repository. It is not a separate Python
package. It reuses `core/`, the selected VCG database, and the resumable
optimizer and storage code from `tuning/`.

Generate and audit the fixed 500-COP dataset before running optimization:

```bash
python final_experiment/generate_dataset.py
python final_experiment/generate_dataset.py --check
```

The generated manifest is stored locally at
`final_experiment/artifacts/manifest.json`. It contains the complete fixed COP
dataset: constraints, QUBO matrices, VCG assignments, PC-QAOA partitions,
penalty values, feasible strings, optimal strings, and dataset metadata. Both
method runners read this file and record its hash in `run.json`. This ensures
that PC-QAOA and PenaltyQAOA use the same COPs and prevents a run from resuming
with a different dataset.

The `tuning/artifacts/` and `final_experiment/artifacts/` directories are local
and are not tracked by Git. A new computer therefore needs a copy of the
selected VCG database and manifest, or must regenerate them before running the
experiment.

Run a short smoke test for either method:

```bash
python final_experiment/run_pcqaoa.py smoke --workers 2 --threads 6
python final_experiment/run_penaltyqaoa.py smoke --workers 2 --threads 6
```

Run or resume the full methods independently. Their output directories do not
overlap, so they may run on different computers from copies of the same commit
and manifest.

```bash
python final_experiment/run_pcqaoa.py run --workers 2 --threads 6
python final_experiment/run_pcqaoa.py resume --workers 2 --threads 6

python final_experiment/run_penaltyqaoa.py run --workers 2 --threads 6
python final_experiment/run_penaltyqaoa.py resume --workers 2 --threads 6
```

Use `--shard-count N --shard-index I` to divide one method across several
machines. Give each machine a separate `--output` directory, then copy the
per-instance task directories into one result directory after the runs stop.
Task files are independent and are identified by the dataset, method,
configuration, and source hashes.

Status commands do not launch optimization:

```bash
python final_experiment/run_pcqaoa.py status
python final_experiment/run_penaltyqaoa.py status
```

Each worker processes one COP and exits, releasing JAX compilation memory.
Optimizer parameters, Adam state, random state, and current progress are saved
every ten steps and at restart and depth boundaries. PC-QAOA and PenaltyQAOA
use matched initialization seeds for the same COP and depth.

Open `final_experiment/notebooks/final_experiment_results.ipynb` to read saved
results and export the analysis figures. The notebook does not start training.
It reads partial or completed runs and saves PDF and PNG figures under
`final_experiment/results/figures/`.
