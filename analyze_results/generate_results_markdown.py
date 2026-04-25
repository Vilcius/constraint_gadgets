"""
generate_results_markdown.py — Generate GitHub-readable markdown tables of raw
experimental results for the disjoint and overlapping problem sets.

Produces:
    results/disjoint/README.md
    results/overlapping/README.md

Each file contains:
  1. Problem definitions table  — cop_id, n_x, constraints (LaTeX math),
     structural constraints, penalized constraints, circuit resources
  2. Results table — one row per (cop_id, method, QAOA layer) with
     AR_feas, P(feas), P(opt); empty cells for non-completed runs

Usage
-----
    cd /home/vilcius/Papers/constraint_gadget/code
    python analyze_results/generate_results_markdown.py
"""
from __future__ import annotations
import ast, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core import constraint_handler as ch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_latex(c: str) -> str:
    """Convert a constraint string to GitHub-renderable LaTeX math."""
    s = c.strip()
    # Replace x_N with x_{N} for multi-digit indices
    s = re.sub(r'x_(\d+)', lambda m: f'x_{{{m.group(1)}}}', s)
    # Replace multiplication x_{i}*x_{j} -> x_{{i}} x_{{j}}
    s = re.sub(r'\*', r' \\cdot ', s)
    # Operators
    s = s.replace('<=', r'\leq').replace('>=', r'\geq')
    s = s.replace('==', '=')
    return f'${s}$'


def _partition(constraints: list[str]):
    parsed = ch.parse_constraints(constraints)
    si, pi = ch.partition_constraints(parsed, strategy='auto')
    return si, pi


def _fmt(val, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return ''
    return f'{val:.{decimals}f}'


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(problem_set: str) -> None:
    """
    problem_set : 'disjoint' or 'overlapping'
    """
    params_path  = f'run/params/experiment_params_{problem_set}.jsonl'
    ar_path      = f'results/{problem_set}/comparison_ar_all_layers.pkl'
    cr_path      = f'results/{problem_set}/circuit_resources.pkl'
    out_path     = f'results/{problem_set}/README.md'

    # ── Load params ──────────────────────────────────────────────────────────
    with open(params_path) as f:
        tasks = [json.loads(l) for l in f]

    # ── Load results (may be missing for non-completed cops) ─────────────────
    ar_df = pd.read_pickle(ar_path) if os.path.exists(ar_path) else pd.DataFrame()
    cr_df = pd.read_pickle(cr_path) if os.path.exists(cr_path) else pd.DataFrame()

    # Build lookup: constraints_hash -> results rows
    results_by_hash: dict[str, pd.DataFrame] = {}
    if not ar_df.empty:
        ar_df['_hash'] = ar_df['constraints_hash']
        for h, grp in ar_df.groupby('_hash'):
            results_by_hash[h] = grp

    # Build lookup: constraints_hash -> circuit resources row
    cr_by_hash: dict[str, dict] = {}
    if not cr_df.empty:
        for _, row in cr_df.iterrows():
            cr_by_hash[row['constraints_hash']] = row.to_dict()

    methods = ['HybridQAOA', 'PenaltyQAOA']
    layers  = [1, 2, 3, 4, 5]

    lines: list[str] = []

    # ── Title ────────────────────────────────────────────────────────────────
    title = problem_set.capitalize()
    lines += [
        f'# {title} Problem Set — Raw Results',
        '',
        f'**{len(tasks)} problem instances** ({problem_set} constraint variable supports).',
        'Empty cells indicate runs that did not complete (transient cluster errors).',
        '',
        '---',
        '',
    ]

    # ── Table 1: Problem definitions ─────────────────────────────────────────
    lines += [
        '## Problem Definitions',
        '',
        '| COP | $n_x$ | Families | Structural constraints | Penalized constraints'
        ' | $n_\\text{qubits}$ (H) | SP gates (H) | Layer gates (H)'
        ' | $n_\\text{qubits}$ (P) | SP gates (P) | Layer gates (P) |',
        '|-----|-------|----------|------------------------|----------------------'
        '-|----------------------|--------------|----------------|'
        '----------------------|--------------|----------------|',
    ]

    for cop_id, task in enumerate(tasks):
        constraints = task['constraints']
        families    = task.get('families', [])
        n_x         = task['n_x']
        h_str       = str(sorted(constraints))  # constraints_hash

        si, pi = _partition(constraints)
        struct_latex  = '<br>'.join(_to_latex(constraints[i]) + f' ({families[i]})' for i in si)
        penalty_latex = '<br>'.join(_to_latex(constraints[i]) + f' ({families[i]})' for i in pi) or '—'

        fam_str = ', '.join(families)

        cr = cr_by_hash.get(h_str, {})
        nq_h  = cr.get('n_qubits_h', '')
        sp_h  = cr.get('sp_total_h', '')
        lay_h = cr.get('layer_total_h', '')
        nq_p  = cr.get('n_qubits_p', '')
        sp_p  = cr.get('sp_total_p', '')
        lay_p = cr.get('layer_total_p', '')

        lines.append(
            f'| {cop_id} | {n_x} | {fam_str} | {struct_latex} | {penalty_latex}'
            f' | {nq_h} | {sp_h} | {lay_h}'
            f' | {nq_p} | {sp_p} | {lay_p} |'
        )

    lines += ['', '---', '']

    # ── Table 2: Results per layer ────────────────────────────────────────────
    lines += [
        '## Results',
        '',
        '`H` = HybridQAOA, `P` = PenaltyQAOA.',
        '',
    ]

    # Header: COP | n_x | method | p=1 AR | p=1 Pfeas | p=1 Popt | p=2 ... | p=5 ...
    layer_headers = ' | '.join(
        f'$p={p}$ AR$_f$ | $p={p}$ $P_f$ | $p={p}$ $P_o$'
        for p in layers
    )
    sep = ' | '.join('--- | --- | ---' for _ in layers)
    lines += [
        f'| COP | $n_x$ | Method | {layer_headers} |',
        f'|-----|-------|--------|{sep}|',
    ]

    for cop_id, task in enumerate(tasks):
        constraints = task['constraints']
        n_x         = task['n_x']
        h_str       = str(sorted(constraints))

        res_grp = results_by_hash.get(h_str)

        for method_short, method_full in [('H', 'HybridQAOA'), ('P', 'PenaltyQAOA')]:
            cells: list[str] = []
            for p in layers:
                if res_grp is not None:
                    row = res_grp[
                        (res_grp['method'] == method_full) &
                        (res_grp['layer'] == p)
                    ]
                    if len(row) > 0:
                        r = row.iloc[0]
                        cells.append(
                            f'{_fmt(r.get("AR_feas"))} | '
                            f'{_fmt(r.get("p_feasible"))} | '
                            f'{_fmt(r.get("p_optimal"))}'
                        )
                        continue
                cells.append(' |  | ')

            lines.append(
                f'| {cop_id} | {n_x} | `{method_short}` | ' +
                ' | '.join(cells) + ' |'
            )

    lines += ['', '---', '']
    lines.append(
        '*Generated by `analyze_results/generate_results_markdown.py`.*'
    )

    os.makedirs(f'results/{problem_set}', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {out_path}  ({len(tasks)} problems, {sum(1 for _ in results_by_hash)} with results)')


if __name__ == '__main__':
    generate('disjoint')
    generate('overlapping')
    print('Done.')
