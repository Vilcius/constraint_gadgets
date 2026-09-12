# Resource-estimation snapshots

These tables and seven PDF/PNG figures summarize the saved tuning constraints.
Study A: 250 gadgets and 50 penalty components, in two gate sets (600 rows).
Study B: 50 constraint occurrences in both representations and two gate sets,
and 100 PenaltyQAOA tasks at depths 1--5 in two gate sets (1200 rows).
23 Study B constraint occurrences lack a saved VCG; they are marked missing,
not assigned zero gates. No training or statevector simulation was run.

All 250 actual saved VCG circuit traces were independently checked against the
NISQ counts. Full PenaltyQAOA counts obey Gprep + p Glayer, and penalty settings
share their circuit topology. `validation.json` records this analysis check;
`*_estimator_policy.json` records code hashes, gate sets and synthesis precision.

The notebook's new resource cells contain executed outputs for viewing without
raw results. To recompute them, use the original saved task directories and
`python tuning/resource_analysis.py --study both`. Notebook execution itself
requires those saved results. These files are snapshots, not optimizer checkpoints.

Component comparisons use one VCG preparation and one merged penalty cost layer.
They are not full PC-QAOA versus PenaltyQAOA comparisons. FTQC counts concern
operation synthesis and do not include error correction or physical hardware.
