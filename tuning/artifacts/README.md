# Selected Study A gadgets

`selected_vcg_db.pkl` contains the 50 gadgets trained with the selected fidelity loss. `selected_loss.json` records the selection and original run identifier; `manifest_a.json` records the constraint dataset.

Each entry stores the constraint, selected angles and depth, legacy parameter layout and slot counts, returned statevector, fidelity, feasibility (`ar`), entropy, and loss name. Reconstruct the feasibility Hamiltonian with `VCG(entry["constraints"])`; no training is needed. The existing `core.pc_qaoa._load_vcg_gadget` loader accepts this database via `db_path`. Preserve the saved legacy parameter layout, including the identity slot, when using the angles.

For penalized-constraint resource estimates, use `core.qaoa_base.build_penalty_hamiltonian` with the existing constraint parser and slack allocation, or construct `PenaltyQAOA(..., compile_on_init=False)` to obtain `penalty_Ham`, `full_Ham`, and wire counts without compiling optimization. The current resource estimator has not been repaired by this export.

Raw optimization checkpoints, restart histories, and saved datasets are tracked in `tuning/results/full` and `tuning/results/full-b-five-methods`. The archived Study A source is tracked in `tuning/results/legacy-a-source`. Notebook defaults and report regeneration read these completed runs directly. Smoke runs, Python caches, and process locks remain ignored.
