"""
generate_results_markdown.py — GitHub-readable markdown tables of raw results.

Produces:
    results/disjoint/README.md
    results/overlapping/README.md

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
# LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_latex(c: str) -> str:
    """Convert a raw constraint string to clean GitHub-renderable LaTeX."""
    s = c.strip()

    # x_N  →  x_{N}
    s = re.sub(r'x_(\d+)', lambda m: f'x_{{{m.group(1)}}}', s)

    # coefficient * x_{i}  →  coeff x_{i}  (no cdot for scalar-variable)
    s = re.sub(r'(\d+)\s*\*\s*(x_\{)', r'\1\2', s)
    # drop leading coefficient of 1 before variable (1x_i → x_i)
    s = re.sub(r'(?<![0-9])1(x_\{)', r'\1', s)

    # x_{i}*x_{i}  →  x_{i}^2  (squared term)
    def _square(m):
        v = m.group(1)
        return f'{v}^2'
    s = re.sub(r'(x_\{\d+\})\*\1', _square, s)

    # x_{i}*x_{j}  →  x_{i}x_{j}  (product, no cdot needed)
    s = s.replace('*', '')

    # operators
    s = s.replace('<=', r'\leq').replace('>=', r'\geq')
    s = s.replace('==', '=')

    return s


def _wrap_latex(c: str, wrap: int = 35) -> str:
    """
    Render constraint as LaTeX, splitting into multiple $...$ chunks at
    '+'/'-' boundaries so no chunk exceeds ~wrap characters.
    Chunks are joined with <br> for use in markdown table cells.
    """
    expr = _to_latex(c)

    # Split on inequality/equality to separate LHS from RHS
    for op in (r'\leq', r'\geq', '='):
        if op in expr:
            idx = expr.index(op)
            lhs = expr[:idx].strip()
            rhs = expr[idx:].strip()
            break
    else:
        lhs, rhs = expr, ''

    # Tokenise LHS at top-level '+' and '-' (keep the sign with the term)
    tokens: list[str] = []
    current = ''
    for ch_c in lhs:
        if ch_c in '+-' and current.strip():
            tokens.append(current)
            current = ch_c
        else:
            current += ch_c
    if current.strip():
        tokens.append(current)

    # Group tokens into lines of ≤ wrap chars
    lines: list[str] = []
    line = ''
    for tok in tokens:
        candidate = (line + tok).strip()
        if len(candidate) > wrap and line:
            lines.append(line.strip())
            line = tok
        else:
            line = candidate
    if line.strip():
        lines.append(line.strip())

    # Attach RHS to final line
    if rhs and lines:
        lines[-1] = lines[-1] + ' ' + rhs
    elif rhs:
        lines.append(rhs)

    return '<br>'.join(f'${ln}$' for ln in lines if ln.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Partition helper
# ─────────────────────────────────────────────────────────────────────────────

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
    params_path = f'run/params/experiment_params_{problem_set}.jsonl'
    ar_path     = f'results/{problem_set}/comparison_ar_all_layers.pkl'
    cr_path     = f'results/{problem_set}/circuit_resources.pkl'
    out_path    = f'results/{problem_set}/README.md'

    with open(params_path) as f:
        tasks = [json.loads(l) for l in f]

    ar_df = pd.read_pickle(ar_path) if os.path.exists(ar_path) else pd.DataFrame()
    cr_df = pd.read_pickle(cr_path) if os.path.exists(cr_path) else pd.DataFrame()

    # Lookup by constraints_hash (stored as str(constraints_list), original order)
    results_by_hash: dict[str, pd.DataFrame] = {}
    if not ar_df.empty:
        for h, grp in ar_df.groupby('constraints_hash'):
            results_by_hash[h] = grp

    cr_by_hash: dict[str, dict] = {}
    if not cr_df.empty:
        for _, row in cr_df.iterrows():
            cr_by_hash[row['constraints_hash']] = row.to_dict()

    methods = ['HybridQAOA', 'PenaltyQAOA']
    layers  = [1, 2, 3, 4, 5]

    lines: list[str] = []
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

    # ── Table 1: Problem definitions ──────────────────────────────────────────
    lines += [
        '## Problem Definitions',
        '',
        '| COP | $n_x$ | Families | Structural constraints | Penalized constraints'
        ' | Qubits (H/P) | SP gates (H/P) | Layer gates (H/P) |',
        '|-----|-------|----------|------------------------|----------------------'
        '-|-------------|---------------|------------------|',
    ]

    for cop_id, task in enumerate(tasks):
        constraints = task['constraints']
        families    = task.get('families', [])
        n_x         = task['n_x']
        h_str       = str(constraints)          # split_results hash (original order)
        h_str_sorted = str(sorted(constraints)) # circuit_resources hash (sorted)

        si, pi = _partition(constraints)
        struct  = '<br>'.join(
            _wrap_latex(constraints[i]) + f' *({families[i]})*' for i in si)
        penalty = '<br>'.join(
            _wrap_latex(constraints[i]) + f' *({families[i]})*' for i in pi) or '—'

        fam_str = ', '.join(families)

        cr = cr_by_hash.get(h_str_sorted, {})
        nq_h  = cr.get('n_qubits_h', '')
        nq_p  = cr.get('n_qubits_p', '')
        sp_h  = cr.get('sp_total_h', '')
        sp_p  = cr.get('sp_total_p', '')
        lay_h = cr.get('layer_total_h', '')
        lay_p = cr.get('layer_total_p', '')

        lines.append(
            f'| {cop_id} | {n_x} | {fam_str} | {struct} | {penalty}'
            f' | {nq_h}/{nq_p} | {sp_h}/{sp_p} | {lay_h}/{lay_p} |'
        )

    lines += ['', '---', '']

    # ── Table 2: Results per layer ────────────────────────────────────────────
    lines += [
        '## Results',
        '',
        '`H` = HybridQAOA, `P` = PenaltyQAOA. '
        'Columns: AR$_f$ = AR$_{\\text{feas}}$, $P_f$ = $P(\\text{feas})$, '
        '$P_o$ = $P(\\text{opt})$.',
        '',
    ]

    p_headers = ' | '.join(
        f'$p={p}$ AR$_f$ | $p={p}$ $P_f$ | $p={p}$ $P_o$' for p in layers)
    p_sep = ' | '.join('--- | --- | ---' for _ in layers)
    lines += [
        f'| COP | $n_x$ | Method | {p_headers} |',
        f'|-----|-------|--------|{p_sep}|',
    ]

    for cop_id, task in enumerate(tasks):
        constraints = task['constraints']
        n_x         = task['n_x']
        h_str       = str(constraints)

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
