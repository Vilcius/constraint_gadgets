# Final PC-QAOA, PenaltyQAOA, and QAOA+ experiment

This folder is part of the existing repository. It is not a separate Python
package. It reuses `core/`, the selected VCG database, and the resumable
optimizer and storage code from `tuning/`.

Generate and audit the fixed 500-COP dataset before running optimization:

```bash
python final_experiment/random_cops/generate_dataset.py
python final_experiment/random_cops/generate_dataset.py --check
```

The fixed dataset is committed at
`final_experiment/random_cops/artifacts/manifest.json`. It contains the constraints,
objectives, reference solutions, VCG assignments, and saved structural partitions.
The selected VCG library is tracked under `tuning/results/full/`.

Completed scientific JSON records, result tables, resource estimates, and figures
are committed for the three notebooks. Checkpoints, process locks, temporary runs,
and intermediate per-depth gate-budget checkpoints remain local. Archived
compiled runs are retained only for the notebook's explicit backend comparison.
The gate-budget runner has no dependency on the separate EV experiment.

```bash
python final_experiment/random_cops/run_pcqaoa.py smoke --workers 2 --threads 6
python final_experiment/random_cops/run_penaltyqaoa.py smoke --workers 2 --threads 6
python final_experiment/random_cops/run_qaoa_plus.py smoke --workers 2 --threads 6
```

Run or resume the full methods independently. Their output directories do not
overlap, so they may run on different computers from copies of the same commit
and manifest.

```bash
python final_experiment/random_cops/run_pcqaoa.py run --workers 2 --threads 6
python final_experiment/random_cops/run_pcqaoa.py resume --workers 2 --threads 6

python final_experiment/random_cops/run_penaltyqaoa.py run --workers 2 --threads 6
python final_experiment/random_cops/run_penaltyqaoa.py resume --workers 2 --threads 6

python final_experiment/random_cops/run_qaoa_plus.py run --workers 2 --threads 6
python final_experiment/random_cops/run_qaoa_plus.py resume --workers 2 --threads 6
```

Use `--shard-count N --shard-index I` to divide one method across several
machines. Give each machine a separate `--output` directory, then copy the
per-instance task directories into one result directory after the runs stop.
Task files are independent and are identified by the dataset, method,
configuration, and source hashes.

Status commands do not launch optimization:

```bash
python final_experiment/random_cops/run_pcqaoa.py status
python final_experiment/random_cops/run_penaltyqaoa.py status
python final_experiment/random_cops/run_qaoa_plus.py status
```

Each worker processes one COP and exits, releasing JAX compilation memory.
Optimizer parameters, Adam state, random state, and current progress are saved
every ten steps and at restart and depth boundaries. All four methods use
matched initialization seeds for the same COP and depth.

The QAOA+ runner enumerates each saved feasible set to construct a joint
feasibility-preserving mixer. Controlled single-bit flips and XY swaps provide
local transitions. If those transitions do not connect the full feasible set,
the constructor adds exact two-state bridge rotations between components. This
ensures that every noiseless QAOA+ circuit remains feasible and that every
feasible state is reachable from the initial state. The construction depends
on complete feasible-set enumeration and is intended for these small benchmark
instances; it is not a scalable mixer-synthesis method.

Open `final_experiment/random_cops/notebooks/final_experiment_results.ipynb` to read saved
results and export the analysis figures. The notebook does not start training.
It reads partial or completed runs and saves PDF and PNG figures under
`final_experiment/random_cops/results/figures/`.

Open `final_experiment/random_cops/notebooks/vcg_results.ipynb` for the VCG loss-selection,
fidelity, training-depth, and constraint-resource figures. The relative link
`final_experiment/random_cops/vcg_results` points to the authoritative completed data in
`tuning/results/full`; the data are not duplicated. Set `PCQAOA_VCG_RESULTS`
to read another VCG result directory.

The final comparison notebook estimates full PC-QAOA, PenaltyQAOA, and QAOA+
circuit resources from the fixed manifest and selected VCG database. It caches the
result under `final_experiment/random_cops/results/resource_estimates/`. The first resource
calculation takes several minutes; later notebook executions read the cache. Build
or resume this cache outside the notebook with:

```bash
python final_experiment/random_cops/compute_resources.py
```

## PC-QAOA With StatePrep

`run_pcqaoa_stateprep.py` replaces only the trained VCG preparation circuits with the exact uniform superposition of each local constraint's feasible bitstrings, constructed with `qml.StatePrep`. Its reversible rotation/CNOT decomposition is prepared once per constraint and reused at every depth. The inherited PC-QAOA circuit applies that preparation and its adjoint in the same Grover mixer. Existing exact gadgets, the saved partition, penalty coefficient, Hamiltonian, slack registers, angle convention, optimizer, and initialization seeds are unchanged.

The full run uses depths 1–5, 10 restarts per depth, 50 Adam steps, learning rate 0.01, and 10,000 final shots. No VCG training is performed. Copy the existing 500-COP manifest to the other computer before launching; the dataset must be identical.

```bash
python final_experiment/random_cops/run_pcqaoa_stateprep.py smoke --workers 2 --threads 6
python final_experiment/random_cops/run_pcqaoa_stateprep.py run --workers 2 --threads 6
python final_experiment/random_cops/run_pcqaoa_stateprep.py resume --workers 2 --threads 6
python final_experiment/random_cops/run_pcqaoa_stateprep.py status
```

Full outputs go to `results/pcqaoa_stateprep/`; smoke outputs go to `results/smoke_pcqaoa_stateprep/`. The comparison notebook reads only the full StatePrep directory by default and draws the variant in purple. Set `PCQAOA_FINAL_STATEPREP_RESULTS` to select another directory. It calculates StatePrep circuit resources for all 500 COPs at depths 1 and 5, counting preparation once and its inverse/preparation twice per gadget mixer per layer. Resource estimates use the separate resumable cache `results/stateprep_resource_estimates/`. Existing three-method resource files remain separate.

To calculate only the StatePrep gate-count table without running the notebook or training:

```bash
python final_experiment/random_cops/compute_resources.py --stateprep-only
```

## Gate-budget comparison

The gate-budget comparison uses QAOA+ at one layer to set deep PC-QAOA depths, with a minimum of one and no depth cap. It does not add another QAOA+ depth-five comparison. See [gate_budget/README.md](gate_budget/README.md) for the uncompiled StatePrep and VCG runners, checkpoint/resume commands, two-machine sharding and inputs. The separate results notebook is [notebooks/gate_budget_results.ipynb](notebooks/gate_budget_results.ipynb). It reads existing targets immediately and does not launch training.

## Uncompiled Four-Method Reruns

Add `--uncompiled` to each original method's runner. These runs read the same manifest and keep depths 1–5, 10 restarts, 50 Adam steps, learning rate 0.01, original JAX PRNG seed derivation, warm starts, final-restart selection and 10,000 shots. All four methods use NumPy statevector evolution and exact adjoint gradients without circuit JIT. Fixed VCG and StatePrep circuits are evaluated once. Their original circuit resource counts are unchanged.

Results are isolated under `results/uncompiled/<method>/`; smoke runs use `results/uncompiled/smoke_<method>/`. The existing compiled results are never overwritten. `resume`, `status`, worker/thread limits, sharding and optimizer checkpoints work as in the original runner. Circuit compilation time is zero. Preparation and probability evaluation have separate timers. NumPy Adam uses the original Optax formulas and JAX initial angles; floating-point differences can still alter optimized solutions. Exact equality of final probabilities is not guaranteed.

From the repository root, with `testWH` active:

```bash
python final_experiment/random_cops/run_pcqaoa.py run --uncompiled --workers 2 --threads 6
python final_experiment/random_cops/run_pcqaoa_stateprep.py run --uncompiled --workers 2 --threads 6
python final_experiment/random_cops/run_penaltyqaoa.py run --uncompiled --workers 2 --threads 6
python final_experiment/random_cops/run_qaoa_plus.py run --uncompiled --workers 2 --threads 6
```

Replace `run` with `smoke` for a reduced test or `resume` to continue. For controlled timing comparisons, use the same hardware and CPU allocations and avoid competing workloads.

The original final-results notebook has two appended timing figures: uncompiled cumulative optimization time through depth five, and paired original-compiled versus uncompiled optimization time. Its original cells, performance data and resource plots are unchanged. The new cells never start optimization. Compiled versus NumPy execution compares implementations, not JIT alone.
