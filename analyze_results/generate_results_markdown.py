"""
generate_results_markdown.py — GitHub-readable markdown tables of raw results.

Produces:
    results/disjoint/README.md
    results/overlapping/README.md

Each file has one table: problem definition + circuit resources + results
for both methods at p=1..5.

Usage
-----
    cd /home/vilcius/Papers/constraint_gadget/code
    python analyze_results/generate_results_markdown.py
"""
from __future__ import annotations
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core import constraint_handler as ch


# ─────────────────────────────────────────────────────────────────────────────
# Unicode math helpers (no LaTeX — renders reliably in GitHub table cells)
# ─────────────────────────────────────────────────────────────────────────────

_SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

def _to_unicode(c: str) -> str:
    """Convert a constraint string to Unicode math for GitHub table display."""
    s = c.strip()

    # x_N  →  xₙ
    s = re.sub(r'x_(\d+)', lambda m: 'x' + m.group(1).translate(_SUB), s)

    # drop coefficient 1 before variable: 1xᵢ → xᵢ  (not preceded by another digit)
    s = re.sub(r'(?<!\d)1(x[₀₁₂₃₄₅₆₇₈₉])', r'\1', s)

    # xᵢ*xᵢ  →  xᵢ²  (squared)
    s = re.sub(r'(x[₀-₉]+)\*(x[₀-₉]+)',
               lambda m: m.group(1) + '²' if m.group(1) == m.group(2)
               else m.group(1) + m.group(2), s)

    # remaining *  →  nothing (implicit multiplication)
    s = s.replace('*', '')

    # operators
    s = s.replace('<=', '≤').replace('>=', '≥').replace('==', '=')

    return s


def _wrap(c: str, wrap: int = 32) -> str:
    """Wrap a constraint at + / - boundaries after ~wrap chars using <br>."""
    expr = _to_unicode(c)

    # Split LHS from operator+RHS
    rhs = ''
    for op in ('≤', '≥', '='):
        if op in expr:
            idx = expr.index(op)
            rhs  = expr[idx:].strip()
            expr = expr[:idx].strip()
            break

    # Tokenise at top-level + and -
    tokens: list[str] = []
    cur = ''
    for ch_c in expr:
        if ch_c in '+-' and cur.strip():
            tokens.append(cur)
            cur = ch_c
        else:
            cur += ch_c
    if cur.strip():
        tokens.append(cur)

    # Group into lines ≤ wrap chars
    lines: list[str] = []
    line = ''
    for tok in tokens:
        cand = (line + tok).strip()
        if len(cand) > wrap and line:
            lines.append(line.strip())
            line = tok
        else:
            line = cand
    if line.strip():
        lines.append(line.strip())

    # Attach RHS to last line
    if rhs and lines:
        lines[-1] += ' ' + rhs
    elif rhs:
        lines.append(rhs)

    return '<br>'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def _partition(constraints: list[str]):
    parsed = ch.parse_constraints(constraints)
    si, pi = ch.partition_constraints(parsed, strategy='auto')
    return si, pi


def _fmt(val, d: int = 3) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return ''
    return f'{val:.{d}f}'


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(problem_set: str) -> None:
    params_path = f'run/params/experiment_params_{problem_set}.jsonl'
    ar_path     = f'results/{problem_set}/comparison_ar_all_layers.pkl'
    cr_path     = f'results/{problem_set}/circuit_resources.pkl'
    out_path    = f'results/{problem_set}/README.md'

    with open(params_path) as f:
        tasks = [json.loads(l) for l in f]

    ar_df = pd.read_pickle(ar_path) if os.path.exists(ar_path) else pd.DataFrame()
    cr_df = pd.read_pickle(cr_path) if os.path.exists(cr_path) else pd.DataFrame()

    # results lookup: key = str(constraints) — original order, matches split_results
    results_by_hash: dict[str, pd.DataFrame] = {}
    if not ar_df.empty:
        for h, grp in ar_df.groupby('constraints_hash'):
            results_by_hash[h] = grp

    # circuit resources lookup: key = str(sorted(constraints))
    cr_by_hash: dict[str, dict] = {}
    if not cr_df.empty:
        for _, row in cr_df.iterrows():
            cr_by_hash[row['constraints_hash']] = row.to_dict()

    layers  = [1, 2, 3, 4, 5]
    methods = [('H', 'HybridQAOA'), ('P', 'PenaltyQAOA')]

    lines: list[str] = []
    title = problem_set.capitalize()
    lines += [
        f'# {title} Problem Set — Raw Results',
        '',
        f'**{len(tasks)} problem instances.** '
        'Empty metric cells = run did not complete (transient cluster errors).',
        '',
        'Column key: **Q** = qubits, **SP** = state-prep gates, '
        '**L** = gates per layer, **AR** = AR_feas, **Pf** = P(feas), **Po** = P(opt).',
        '',
    ]

    # ── Build header ──────────────────────────────────────────────────────────
    p_headers = ' | '.join(
        f'p={p} AR | p={p} Pf | p={p} Po' for p in layers
    )
    p_sep = ' | '.join(':---: | :---: | :---:' for _ in layers)

    lines += [
        f'| COP | nx | Method | Structural constraints | Penalized constraints'
        f' | Q | SP | L | {p_headers} |',
        f'|:---:|:--:|:------:|------------------------|---------------------'
        f'-|:-:|:--:|:-:|{p_sep}|',
    ]

    # ── One row per (cop, method) ─────────────────────────────────────────────
    for cop_id, task in enumerate(tasks):
        constraints  = task['constraints']
        families     = task.get('families', [])
        n_x          = task['n_x']
        h_str        = str(constraints)
        h_str_sorted = str(sorted(constraints))

        si, pi = _partition(constraints)
        struct  = '<br>'.join(
            _wrap(constraints[i]) + f' *({families[i]})*' for i in si)
        penalty = '<br>'.join(
            _wrap(constraints[i]) + f' *({families[i]})*' for i in pi) or '—'

        res_grp = results_by_hash.get(h_str)
        cr = cr_by_hash.get(h_str_sorted, {})

        for method_short, method_full in methods:
            nq  = cr.get(f'n_qubits_{"h" if method_short == "H" else "p"}', '')
            sp  = cr.get(f'sp_total_{"h" if method_short == "H" else "p"}', '')
            lay = cr.get(f'layer_total_{"h" if method_short == "H" else "p"}', '')

            metric_cells: list[str] = []
            for p in layers:
                if res_grp is not None:
                    row = res_grp[
                        (res_grp['method'] == method_full) &
                        (res_grp['layer'] == p)
                    ]
                    if len(row) > 0:
                        r = row.iloc[0]
                        metric_cells.append(
                            f'{_fmt(r.get("AR_feas"))} | '
                            f'{_fmt(r.get("p_feasible"))} | '
                            f'{_fmt(r.get("p_optimal"))}'
                        )
                        continue
                metric_cells.append(' | | ')

            metrics_str = ' | '.join(metric_cells)

            # Only print constraint columns on first method row; blank for second
            if method_short == 'H':
                struct_cell  = struct
                penalty_cell = penalty
            else:
                struct_cell  = '↑'
                penalty_cell = '↑'

            lines.append(
                f'| {cop_id} | {n_x} | `{method_short}` '
                f'| {struct_cell} | {penalty_cell} '
                f'| {nq} | {sp} | {lay} '
                f'| {metrics_str} |'
            )

    lines += ['', '---', '']
    lines.append('*Generated by `analyze_results/generate_results_markdown.py`.*')

    os.makedirs(f'results/{problem_set}', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    n_with = sum(1 for t in tasks if str(t['constraints']) in results_by_hash)
    print(f'Wrote {out_path}  ({len(tasks)} problems, {n_with} with results)')


if __name__ == '__main__':
    generate('disjoint')
    generate('overlapping')
    print('Done.')
