# Loss and penalty tuning

Run these commands from the **code-tuning worktree root**, not the original
`code/` checkout. This is a scripts folder, not an installable package. It
imports the repository's existing `core/`, `data/`, and `run/` code.

## Setup and first checks

Use your existing Python environment (Python 3.12 or 3.13):

```bash
python -m pip install -r tuning/requirements.txt
python -m pytest tuning/tests -q
python tuning/run.py smoke --workers 2 --threads 6
python tuning/run.py status --profile smoke
```

`preflight` is an alias for the smoke profile. It checks both studies with
short budgets, keeping all five losses and five penalty methods. Smoke uses
four support-four VCGs and two four-variable COPs; two restarts/two steps and
depth two. It **does not estimate full-study runtime reliably**, nor does its
selected winner constitute scientific evidence. Compilation and optimizer
times and peak worker memory are recorded in `execution_summary.json`.

The default machine allocation is two workers with six physical cores each.
CPU affinity and numerical-library thread limits avoid overlapping worker
allocations. Change `--workers` and `--threads` without invalidating saved
scientific results. A worker handles all settings of one instance, then exits
to release JAX compilation memory.

## Full studies

The configuration is `tuning/config.json`. No command in setup or the notebook
starts full training automatically.

```bash
# Generate and inspect the fixed datasets before training.
python tuning/run.py generate --study b

# These can be launched separately. Do not launch simultaneous commands into
# the same output directory; each command already uses multiple workers.
python tuning/results/legacy-a-source/tuning/run.py resume --study a --output "$PWD/tuning/results/full"
python tuning/run.py run --study b

# Resume or inspect the revised Study B.
python tuning/run.py resume --study b
python tuning/run.py status --study b
python tuning/run.py export --study b
```

Study A uses `tuning/results/full`. Revised Study B (`--study b`) uses
`tuning/results/full-b-five-methods`. New smoke runs use
`tuning/results/smoke-five-methods`, or `smoke-b-five-methods` with `--study b`. Use `--output PATH` for another run; specify the same
profile/configuration when resuming it. `status` and `export` read settings
from the chosen output directory. Commands are independent of current
directory when invoked using an absolute path to `run.py`.

### Study A

- 50 constraints: ten per support size 4–8, two or three per family per size.
  Total family counts are 13 positive inequalities, 13 mixed inequalities,
  12 weighted equalities, and 12 quadratic inequalities.
- Positive and quadratic candidates call the original file-generating
  functions in a temporary directory. Coefficients/capacities therefore
  follow the **generator source**, not the historically different checked-in
  constraint CSVs. Mixed signs and weighted equalities are small extensions.
- Exactly preparable and empty/full/duplicate feasible sets are excluded.
  There is no feasible-fraction cutoff or general factorization filter.
- Five losses: feasibility energy, total infidelity, and conditional-fidelity
  weights 0.5, 1 and 2. All five are eligible for selection.
- Standard-QAOA warm-up: 5 restarts × 150 steps. ma-QAOA depths 1–8:
  20 restarts × 200 steps each, Adam learning rate 0.05.
- At each depth choose minimum final training loss across restarts; stop on
  that state's total fidelity >= 0.999. Return the highest-fidelity depth
  winner. Choose the loss by mean returned fidelity over all 50 constraints.

The existing phase/RX angle conventions and parameter layout are preserved,
including the original VCG identity parameter slot. Initial random draws are
matched across losses, but later inherited angles naturally differ. The
legacy entropy selection is not used by the new tuning workflow.
Entropy is the Shannon entropy of the conditional feasible-state probabilities,
normalized by the log of the feasible-state count. Coverage is the fraction
of feasible states with conditional probability at least 0.001 times the
uniform target probability. Both are diagnostics; neither selects the loss.
As in the original VCG trainer, every restart at a later depth appends random
new-layer angles. PenaltyQAOA retains its different convention: zero new-layer
angles on the first restart, random new-layer angles on subsequent restarts.

### Study B

- 20 fresh calibration COPs, four per size 4–8; two disjoint and two
  overlapping, each containing two or three constraints.
- Candidate supports and constraints use `generate_cops` and its existing
  six-family CSV pools. Candidate selection favors underrepresented families,
  constraint counts and resulting partition counts. The fixed pools and
  random objective generator intentionally have separate provenance.
- Objectives are freshly drawn with the original upper-triangular generator,
  coefficients -5 through 4. PC-QAOA partition/slack counts are descriptive;
  **no PC-QAOA or VCG training occurs in Study B**.
- Five fixed methods: coefficient range, feasible-solution bound, Verma–Lewis
  signed local change, absolute local change, and maximum coefficient.
  No multiplier sweep and no manuscript-reference setting: 100 tasks total.
- The feasible point comes from a seeded random ordering searched without
  replacement, stopping at the first feasible string. No objective values or
  optimal-solution labels enter this search. Its seed, number of checks and
  runtime are saved. Enumeration is suitable for these small COPs; this does
  not establish scalable feasibility search on larger instances.
- Standard QAOA, depths 1–5; 10 restarts × 50 steps per depth, learning rate
  0.01. Restarts minimize penalized expectation and depths are warm-started.
- Select by mean **sampled** P(opt) at depth five. All optima are counted.
  All five methods are selectable; exact probabilities are diagnostic.

Final measurement counts are seeded multinomial draws of 10,000 shots from
the exact PennyLane **decision-register marginal**. This is distributionally
equivalent to measuring all wires and discarding slack bits; it avoids a
second circuit simulation. Seeds are saved. No shot noise enters gradients.
Exact optima label these small calibration problems but do not enter the
five penalty formulas. Range and feasible-solution sufficiency use the integer
residual lower bound of one. Both local rules and maximum coefficient are
heuristics for general overlapping constraints.

## Cancellation, failures and results

Ctrl-C or SIGTERM requests a checkpoint at the next optimizer boundary. A
long JIT compilation cannot be checkpointed halfway through. An abrupt kill
can lose work since the last ten-step checkpoint; parameters, Adam moments,
RNG state and progress are restored afterward. Compilation may repeat.

Each instance/condition owns a separate directory with status, an atomic
checkpoint and its previous valid version, completed-depth records, and its
final result. Histories retain the first step, checkpoint steps, and final
step of each restart to limit repeated checkpoint I/O. Checkpoints are local
Python pickles: load only your own files.
Corrupt current checkpoints fall back to their valid predecessors. Two
commands cannot write concurrently to the same run directory.

Exceptions save tracebacks and the command exits nonzero. Finishing below
the fidelity threshold is a valid nonconverged result. Resume skips finished
tasks and retries unfinished ones. Source/configuration/environment changes
require a new output directory; they are never merged silently.

Final selections require the complete expected set of results. Export writes
summary/depth CSVs, rankings, `selected_loss.json`, `selected_vcg_db.pkl`, and
`selected_penalty.json` where the corresponding study is complete. The VCG
database uses the normalized constraint keys and angle layout expected by
the existing loader. It contains one winning-loss gadget per constraint.
The original gadget database is never overwritten.

Keep these calibration COP IDs out of the later main comparison. Preserve
`run.json` and both manifests with archived results: they include source
hashes, base commit, seeds, coefficients, fixed reference solutions and
installed library versions. Dataset order is fixed before optimization.

## Notebook

Open `tuning/notebooks/tuning_results.ipynb` with the same Python environment.
Set `RESULTS_DIR` in its first code cell, or set `PCQAOA_TUNING_RESULTS`.
It reads existing Study A results from `full` and revised Study B results
from `full-b-five-methods`. Set `PCQAOA_TUNING_B_RESULTS` or `B_RESULTS_DIR`
to read a different Study B run. For a combined smoke run, set both paths to
that smoke directory. Figures use discrete method comparisons, not multipliers.

The notebook reads result files directly, including completed depths of
interrupted tasks. It does not train or modify selections. Mean plots use pointwise 95% Student-t confidence intervals, computed as
mean ± t(0.975, n−1) × sample SD / sqrt(n) across constraints or COPs.
Optimizer restarts are not independent instances. Groups with fewer than two
instances have no interval. Intervals are not clipped to probability bounds.
There is no bootstrap or multiple-comparison adjustment. Small or skewed
groups can have unreliable coverage. Individual-observation plots have no
mean intervals; explicitly named SD columns in tables remain descriptive.
Depth-wise VCG plots include only tasks that actually reached each depth;
fidelity stopping can change the contributing set. The loss ranking instead
uses one returned state from every constraint, including early-stopped tasks.

Figures use manuscript colors, PGF/pdflatex Computer Modern text and PDF/PNG
exports. LaTeX failure falls back to built-in Computer Modern math plus
DejaVu Serif and is recorded in `figures/font_status.json`. Compilation time
includes disposable JIT warm-up calls and dispatch overhead on reused
functions; optimizer time measures synchronized update execution. Peak RSS
is a process high-water mark, not additive per-task memory.

## Verification on this machine

The completed smoke run contains 20 VCG tasks and 20 penalty tasks, including
an overlapping COP with 14 total qubits. The notebook executed on those saved
results and exported 22 PDF/PNG figure pairs using actual LaTeX Computer Modern.
Tests also force LaTeX failure to verify the built-in font fallback.
The original 23 checks passed, including optimizer interruption/resume, sampled ranking,
checkpoint recovery and concurrent-write protection. All 20 smoke gadgets'
saved angles independently reproduce their saved states and quality metrics.

For Study B, combining repeated commuting Pauli terms before PennyLane/JAX
tracing reduced total recorded compilation/warm-up time from about 12,644
seconds to 32 seconds, and peak worker RSS from 10,594 MiB to 1,268 MiB in
these smoke runs. Study A recorded about 8 seconds of compilation/warm-up and
473 MiB peak RSS. These are local measurements, not full-study estimates.
Tests compare the optimized standard-QAOA probabilities, expectations and
gradients against the original circuits, including remapped quadratic terms.

The full datasets are generated but untrained. PenaltyQAOA instances span
4–19 total qubits; the 19-qubit instance has 12 slack qubits. Full-depth
backpropagation can need substantially more memory than the smoke run.
Saved manifests expose these counts so allocation can be reviewed before
starting full training. No tensor-network simulator has been introduced.

## Existing Study A Run and Revised Study B

Study A was already running when Study B changed. Its output files and
scientific settings were left intact. Its workers can finish normally.
The original source was preserved in `tuning/results/legacy-a-source` so that
its strict source/configuration check can still be satisfied if interrupted.
To resume that original run, use the same Python environment as before:

```bash
python tuning/results/legacy-a-source/tuning/run.py resume --study a --output "$PWD/tuning/results/full"
```

Launch revised Study B with `python tuning/run.py run --study b`. Do not use
old Study B checkpoints with the five-method configuration. The new run has
its own provenance and manifest; old results are still readable. Separate
output directories allow separate commands, but each command allocates its
own workers. Let Study A finish before starting the default two-worker B run
to avoid CPU and memory contention.

The notebook formulas describe the five new methods. Historical smoke
measurements above refer to the earlier ten-setting implementation. Revised
Study B smoke results and timings are stored in `smoke-b-five-methods`.

The revised five-method implementation passes 29 tests, including all five
selection candidates, optimum-free penalty inputs, seeded feasibility search,
and rejection of multiplier configurations.

## Collaborator handoff

The selected 50 VCGs are available in `tuning/artifacts/selected_vcg_db.pkl`; see `tuning/artifacts/README.md` for reconstruction and resource-estimation entry points. Completed raw results are tracked in `tuning/results/full` and `tuning/results/full-b-five-methods`, with the original Study A source in `tuning/results/legacy-a-source`. The notebook and report can read these results on a fresh checkout. Historical checkpoints retain their original source/environment identifiers; use the archived source and matching environment if resuming them. Start fresh studies in a new output directory, such as `python tuning/run.py run --study a --output tuning/results/new-a`.

## Resource estimates from completed tuning runs

The tuning branch includes main's qre estimator rewrite (`ab23dce`). It works
with the installed PennyLane 0.43.1; no dependency upgrade is needed.

```bash
python tuning/resource_analysis.py --study both
```

Use `--a-output`, `--b-output`, and `--vcg-db` to read other saved datasets.
This traces saved circuits without training, JAX compilation, or statevector
simulation. Per-instance content-identified JSON caches and CSV tables are
written under each study's `resource_estimates/` directory. Cache identifiers
include source hashes, saved inputs, gate sets, PennyLane version, and the
rotation-synthesis configuration. Interrupted analysis can be rerun.

The notebook has resource tables and figures inside each study. Study A
includes all 250 saved gadgets and 50 isolated penalty components. Study B
includes gadget/penalty components for all 50 constraint occurrences and
full PenaltyQAOA estimates at depths 1--5 for all five penalty settings.
Missing VCGs are explicit and never trigger training. The database may be
replaced to fill these entries. Full PC-QAOA comparisons are deferred while
some Study B VCGs are unavailable.

The estimator uses NISQ and FTQC gate sets defined in
`core/resource_estimation.py`. FTQC synthesis precision is per rotation,
not a total-circuit error budget; these estimates do not include error
correction or physical hardware mapping. Reported gate counts are operation
decompositions, with no angle pruning or global transpilation. Register
qubits and synthesis ancillas are reported separately.

Penalty components use merged Pauli words for shared-angle QAOA. Full Study B
counts match the fixed merged topology in `PenaltyQAOA.tuning_functions`,
including terms with zero angle at a particular penalty setting. Gadget
preparation and one isolated penalty cost layer are component estimates,
not interchangeable full algorithm costs. Full totals include objective
terms, initial Hadamards on all PenaltyQAOA wires, and mixers. Grover mixers
also use gadget unpreparation and repreparation every outer layer. Sums of
isolated constraint components are not upper bounds on full-circuit costs.

The integration also corrects the XY-mixer mirror to trace the actual
`IsingXY` decomposition, including basis-change gates. Resource tests compare
traces with executable circuits and verify missing-gadget and cache behavior.
Final exported resource tables and figure previews are included under
`tuning/artifacts/` for viewing on a fresh checkout.

## Tracked experimental data

The completed Study A and revised Study B directories include datasets, run provenance, optimizer checkpoints and backups, depth/restart histories, saved gadgets, final summaries, selections, resource estimates, and figures. The archived Study A source is tracked too. These files occupy about 105 MiB before Git compression. Smoke runs, new output directories, Python caches, and process locks remain ignored. Preserve original run records rather than mixing historical checkpoints with changed code or settings.
