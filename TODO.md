# TODO

## 1. Feasibility-conditioned AR metric
Add `AR_feas` using the coauthor's formula:

    AR_feas = (f_max_F - E[f(x) : x ∈ F]) / (f_max_F - f*)

where F is the feasible set, f* is the true optimum, f_max_F is the worst
feasible QUBO value, and the expectation is over feasible shots only.
Implemented in `analyze_results/metrics.py` as `ar_feasibility_conditioned`.
Remaining work:
- Add AR_feas to the analysis pipeline (main_analysis.py, plot_ar.py)
- Add AR_feas column to comparison summary CSVs
- Add AR_feas plots alongside existing AR plots

Note: AR_feas ignores P(feas) — a method with one lucky feasible shot gets
AR_feas=1.0. Always report alongside P(feas) and P(opt).

## 2. P(feas) vs layer and P(opt) vs layer figures
Current figures show metrics vs n_x. The text discusses how solution quality
improves with layers, but there are no figures showing this. Add plots of
P(feas) and P(opt) as a function of QAOA depth p.

## 3. Principled penalty weight assignment
Currently uses a single scalar `penalty_weight = 5 + 2*|f_min|` for all
constraints. Goal: find the smallest delta per constraint such that all
infeasible solutions have higher cost than the best feasible solution + 1
(tight lower bound).

## 4. Increase optimization budget
Current settings: HYBRID_STEPS=50, PENALTY_STEPS=50, num_restarts=10.
P(feas) for HybridQAOA is only ~0.30 on average; some constraint types reach
P(feas)=0.0. Increase to STEPS=150+, restarts=20+, and raise VCG ar_threshold
to 0.999 (already set) but also increase gadget budget to ensure gadgets
really reach threshold before the full solver runs.

## 5. Update Conclusion section
The conclusion still describes the old single/two-constraint framing. Update
to reflect: multi-constraint hybrid solver, VCG comparison, Appendix B,
and the improved warm-start results.

## 6. Hamiltonian normalization for PenaltyQAOA
PenaltyQAOA's eigenvalue range is ~3000–4000× larger than HybridQAOA's due
to penalty terms. This makes the QAOA optimization landscape harder (sharper
gradients) and makes the raw AR metric incomparable (addressed by AR_feas).
Fix: rescale the full Hamiltonian to [-1, 1] before optimization so that
step size and convergence budget are comparable across methods. Implement
in `core/penalty_qaoa.py`.
