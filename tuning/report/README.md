# Completed Tuning Report

Open `summary.pdf` for the complete report. `summary.tex` is the standalone
LaTeX wrapper; `tuning_section.tex` contains the manuscript-style main text.
`reproducibility.tex` contains the implementation appendix and extra figures.
All figures and tables use completed results: 250 Study A tasks and 100 Study B tasks.

To rebuild from the original saved results, run from the code-tuning root:

```bash
python tuning/report/build_report.py
```

This reads existing results, regenerates report assets and compiles LaTeX.
It does not train circuits or change the selected methods. To compile this
folder independently of the repository, run `pdflatex summary.tex` twice.
The PDF, LaTeX sources, figures, tables and data snapshots can be moved together.

For manuscript integration, use the text in `tuning_section.tex`, copy its
referenced figures/tables, and adjust asset paths. It requires amsmath,
amssymb, graphicx, booktabs, subcaption and placeins. The standalone wrapper
contains the two bibliography entries; merge them into the manuscript's
bibliography when transferring the section. Labels use the `cal:` prefix.
The report does not change the current manuscript.

The 95% intervals are pointwise Student-t intervals across instances. Paired
penalty comparisons are exploratory and do not establish superiority of
the selected rule. The report distinguishes calibration from held-out
algorithm evaluation and actual execution from hypothetical PC-QAOA counts.

Edit `tuning_section.tex` for the main report text and compile `summary.tex`.
Study B method labels use the same delta symbols in the text, tables and figures.
