"""
diagnose_cases.py -- Deep diagnostic for Cases 3 and 4 (P(opt)=0 mystery).

For each case, runs VCG at p=1 and shows:
  1. QUBO value for every feasible state (which is "optimal"?)
  2. Full probability distribution coloured by outcome type:
       gold   = globally optimal feasible state
       foam   = feasible, sub-optimal
       love   = infeasible
  3. Prints per-feasible-state probability so we can see exactly
     where mass is going.

Run from project root:
    python progress/diagnose_cases.py
"""

import sys, os, itertools, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pennylane import numpy as np
import pennylane as qml

from core import constraint_handler as ch
from core import dicke_state_prep as dsp
from core import qaoa_base as base
from core.vcg import VCG
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results.metrics import feasibility_check, evaluate_qubo
from analyze_results import plot_utils as pu

os.makedirs('progress/figures', exist_ok=True)

SHOTS   = 50_000
STEPS   = 50
RESTARTS = 10
LR      = 0.01

NF_QAOA_RESTARTS = 5
NF_QAOA_STEPS    = 150
NF_MA_RESTARTS   = 10
NF_MA_STEPS      = 100
NF_LR            = 0.05

qubos = read_qubos_from_file('qubos.csv', results_dir='data/')

FOCUS_CASES = [
    {
        'name': 'Case 3 — cardinality+knapsack',
        'constraints': ['x_0 + x_1 >= 2', '5*x_2 + 2*x_3 + 5*x_4 + 5*x_5 <= 9'],
        'n_x': 6, 'qubo_idx': 6,
    },
    {
        'name': 'Case 4 — cardinality+cardinality',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_3 + x_4 >= 1'],
        'n_x': 5, 'qubo_idx': 3,
    },
]

C = pu._ROSE_PINE
COL_OPT   = C['gold']    # globally optimal
COL_FEAS  = C['foam']    # feasible, sub-optimal
COL_INFEAS = C['love']   # infeasible

pu.setup_style()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Probability distributions — VCG p=1\n'
             '(gold = optimal, teal = feasible, red = infeasible)',
             fontsize=12)

for ax, task in zip(axes, FOCUS_CASES):
    name = task['name']
    n_x  = task['n_x']
    Q    = qubos[n_x][task['qubo_idx']]['Q']
    constraints = task['constraints']
    parsed      = ch.parse_constraints(constraints)
    str_idx, pen_idx = ch.partition_constraints(parsed, strategy='auto')
    min_val, optimal_x, total_min = get_optimal_x(Q, constraints)
    penalty_weight = float(5 + 2 * abs(total_min))

    print('=' * 65)
    print(name)
    print(f'  n_x={n_x}, opt={min_val:.3f}, pen_idx={pen_idx}')
    print(f'  optimal_x = {optimal_x}')

    # ── QUBO landscape over all feasible states ───────────────────────
    print('\n  QUBO values over all feasible states:')
    feasible_states = []
    for bits in itertools.product([0, 1], repeat=n_x):
        bs = ''.join(map(str, bits))
        if feasibility_check(bs, constraints, n_x):
            val = evaluate_qubo(bs, Q, n_x)
            is_opt = bs in set(optimal_x)
            feasible_states.append((bs, val, is_opt))

    feasible_states.sort(key=lambda x: x[1])
    for bs, val, is_opt in feasible_states:
        mark = '  ← OPTIMAL' if is_opt else ''
        print(f'    {bs}  QUBO={val:.3f}{mark}')

    # ── Classify non-Dicke/LEQ/Flow structural constraints → VCG ──
    dicke_idxs  = [i for i in str_idx if ch.is_dicke_compatible(parsed[i])]
    leq_idxs    = [i for i in str_idx if ch.is_cardinality_leq_compatible(parsed[i])]
    flow_idxs   = [i for i in str_idx if ch.is_flow_compatible(parsed[i])]
    gadget_idxs = [i for i in str_idx
                   if not ch.is_dicke_compatible(parsed[i])
                   and not ch.is_cardinality_leq_compatible(parsed[i])
                   and not ch.is_flow_compatible(parsed[i])]

    dicke_preps = [dsp.from_parsed_constraint(parsed[i]) for i in dicke_idxs]
    leq_preps   = [dsp.from_cardinality_leq_constraint(parsed[i]) for i in leq_idxs]
    flow_preps  = [dsp.from_flow_constraint(parsed[i]) for i in flow_idxs]

    nf_gadgets = []
    for i in gadget_idxs:
        raw = parsed[i].raw
        print(f'\n  Training VCG: {raw}')
        vcg_nf = VCG(
            constraints=[raw],
            ar_threshold=0.999, max_layers=8,
            qaoa_restarts=NF_QAOA_RESTARTS, qaoa_steps=NF_QAOA_STEPS,
            ma_restarts=NF_MA_RESTARTS,     ma_steps=NF_MA_STEPS,
            lr=NF_LR, samples=SHOTS,
        )
        vcg_nf.train(verbose=False)
        print(f'    AR={vcg_nf.ar:.4f}  layers={vcg_nf.n_layers}  '
              f'single={vcg_nf._single_feasible_bitstring is not None}')
        if vcg_nf._single_feasible_bitstring:
            print(f'    Single state: |{vcg_nf._single_feasible_bitstring}>')
        nf_gadgets.append(vcg_nf)

    state_prep = dicke_preps + leq_preps + flow_preps + nf_gadgets

    # No penalty constraints → cost is just QUBO
    x_wires  = list(range(n_x))
    all_wires = x_wires
    H_qubo   = base.build_qubo_hamiltonian(Q, x_wires)
    H_prob   = H_qubo

    num_gamma = base.count_gamma_terms(H_prob)
    num_beta  = 1

    # ── Optimise p=1 ──────────────────────────────────────────────────
    def cost_fn(angles):
        @qml.qnode(qml.device('lightning.qubit', wires=all_wires))
        def circ(angles):
            for prep in state_prep:
                prep.opt_circuit()
            gammas, betas = base.split_angles(angles, num_gamma, num_beta, 'ma-QAOA')
            base.apply_cost_unitary(H_prob, gammas, 0)
            base.apply_grover_mixer(betas[0][0], all_wires, state_prep)
            return qml.expval(H_prob)
        return circ(angles)

    opt_cost, opt_angles, _ = base.run_optimization(
        cost_fn=cost_fn, n_layers=1,
        num_gamma=num_gamma, num_beta=num_beta,
        angle_strategy='ma-QAOA',
        steps=STEPS, num_restarts=RESTARTS, learning_rate=LR,
    )
    print(f'\n  p=1 opt_cost={opt_cost:.4f}')

    # ── Sample distribution ───────────────────────────────────────────
    @qml.qnode(qml.device('lightning.qubit', wires=all_wires, shots=SHOTS))
    def counts_circ():
        for prep in state_prep:
            prep.opt_circuit()
        gammas, betas = base.split_angles(opt_angles, num_gamma, num_beta, 'ma-QAOA')
        base.apply_cost_unitary(H_prob, gammas, 0)
        base.apply_grover_mixer(betas[0][0], all_wires, state_prep)
        return qml.counts(all_outcomes=True)

    counts = counts_circ()
    total  = sum(counts.values())
    opt_set = set(optimal_x)

    print('\n  Probability per feasible state (descending):')
    feas_probs = []
    for bs, val, is_opt in feasible_states:
        p = counts.get(bs, 0) / total
        feas_probs.append((bs, val, is_opt, p))
    for bs, val, is_opt, p in sorted(feas_probs, key=lambda x: -x[3]):
        mark = '  ← OPTIMAL' if is_opt else ''
        print(f'    {bs}  QUBO={val:.3f}  P={p:.4f}{mark}')

    p_feas = sum(x[3] for x in feas_probs)
    p_opt  = sum(x[3] for x in feas_probs if x[2])
    print(f'\n  P(feas)={p_feas:.4f}  P(opt)={p_opt:.4f}')

    # ── Plot ──────────────────────────────────────────────────────────
    all_states = [format(i, f'0{n_x}b') for i in range(2 ** n_x)]
    probs  = [counts.get(s, 0) / total for s in all_states]
    colors = []
    for s in all_states:
        if s in opt_set:
            colors.append(COL_OPT)
        elif feasibility_check(s, constraints, n_x):
            colors.append(COL_FEAS)
        else:
            colors.append(COL_INFEAS)

    ax.bar(range(len(all_states)), probs, color=colors, width=1.0, linewidth=0)
    uniform_feas = 1.0 / len(feasible_states) if feasible_states else 0
    ax.axhline(uniform_feas, color=C['muted'], linestyle='--', linewidth=1.2,
               label=f'Uniform over feasible (1/{len(feasible_states)})')
    ax.set_title(f'{name}  |  P(feas)={p_feas:.3f}  P(opt)={p_opt:.4f}',
                 fontsize=10)
    ax.set_xlabel('Computational basis state (integer index)')
    ax.set_ylabel('Probability')
    ax.legend(fontsize=8)

patches = [
    mpatches.Patch(color=COL_OPT,   label='Optimal feasible'),
    mpatches.Patch(color=COL_FEAS,  label='Feasible (sub-optimal)'),
    mpatches.Patch(color=COL_INFEAS, label='Infeasible'),
]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = 'progress/figures/diagnose_distributions.png'
pu.save_fig(fig, out)
print(f'\nSaved figure to {out}')
