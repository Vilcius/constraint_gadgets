# TODO

## 1. Feasibility-conditioned AR metric
Compute AR restricted to feasible shots only. Makes HybridQAOA and PenaltyQAOA
comparable on the same footing — currently AR is not comparable because
PenaltyQAOA inflates C_max by ~3000–4000 via penalty terms vs ~60–100 for
HybridQAOA, making its AR appear artificially high.

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
P(feas) for HybridQAOA is only ~0.30. Increase to STEPS=150+, restarts=20+,
and raise VCG ar_threshold to 0.999 (already set) but also increase gadget
budget to ensure it really reaches threshold before the full solver runs.

## 5. Update Conclusion section
The conclusion still describes the old single/two-constraint framing. Update
to reflect: multi-constraint hybrid solver, VCGNoFlag comparison, Appendix B,
and the improved warm-start results.
