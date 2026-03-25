"""
analyse_landscape.py  --  Quantify β-gradient in p=1 Grover landscape.

The key question: is the landscape flat *in the β direction* for Cases 4-6?

For each case, fix γ at its "best" value (the column where E is minimised)
and plot E(β) along that slice.  Also compute the β-range ΔE_β (variation
along β at fixed best-γ) vs the γ-range ΔE_γ (variation along γ at fixed β=0).
A small ΔE_β / ΔE_γ ratio means β is nearly useless (flat in the mixer direction).
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core import constraint_handler as ch
from core import dicke_state_prep as dsp
from core import qaoa_base as base
from core.vcg import VCG
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results import plot_utils as pu

os.makedirs('progress/figures', exist_ok=True)

FOCUS_CASES = [
    {
        'case_num': 1, 'works': True,
        'constraints': ['x_0*x_1 == 0', '2*x_2 + 1*x_3 + 4*x_4 <= 2', 'x_5 + x_6 == 2'],
        'n_x': 7, 'qubo_idx': 1,
    },
    {
        'case_num': 2, 'works': True,
        'constraints': ['3*x_0 + 2*x_1 <= 2', '2*x_2 + 1*x_3 <= 2', 'x_4 + x_5 + x_6 >= 3'],
        'n_x': 7, 'qubo_idx': 8,
    },
    {
        'case_num': 3, 'works': True,
        'constraints': ['x_0 + x_1 >= 2', '5*x_2 + 2*x_3 + 5*x_4 + 5*x_5 <= 9'],
        'n_x': 6, 'qubo_idx': 6,
    },
    {
        'case_num': 4, 'works': False,
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_3 + x_4 >= 1'],
        'n_x': 5, 'qubo_idx': 3,
    },
    {
        'case_num': 5, 'works': False,
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_0 + x_3 + x_4 == 1', 'x_5 + x_6 >= 1'],
        'n_x': 7, 'qubo_idx': 7,
    },
    {
        'case_num': 6, 'works': False,
        'constraints': ['3*x_0 + 2*x_1 <= 2', 'x_1 + x_2 + x_3 <= 2', 'x_4 + x_5 + x_6 == 2'],
        'n_x': 7, 'qubo_idx': 1,
    },
]

N_PTS  = 80
GAMMA_MAX = np.pi
BETA_MAX  = np.pi / 2

NF_THRESHOLD  = 0.999
NF_QAOA_REST  = 5
NF_QAOA_STEPS = 100
NF_MA_REST    = 10
NF_MA_STEPS   = 80
NF_LR         = 0.05
NF_SHOTS      = 10_000

qubos = read_qubos_from_file('qubos.csv', results_dir='data/')


def build_circuit_fn(constraints, n_x, Q):
    parsed    = ch.parse_constraints(constraints)
    str_idx, pen_idx = ch.partition_constraints(parsed, strategy='auto')

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
        vcg_nf = VCG(
            constraints=[raw],
            ar_threshold=NF_THRESHOLD, max_layers=8,
            qaoa_restarts=NF_QAOA_REST, qaoa_steps=NF_QAOA_STEPS,
            ma_restarts=NF_MA_REST,     ma_steps=NF_MA_STEPS,
            lr=NF_LR, samples=NF_SHOTS,
        )
        vcg_nf.train(verbose=False)
        nf_gadgets.append(vcg_nf)

    state_prep = dicke_preps + leq_preps + flow_preps + nf_gadgets

    x_wires = list(range(n_x))
    H_qubo  = base.build_qubo_hamiltonian(Q, x_wires)

    slack_wires = []
    if pen_idx:
        pen_parsed  = [parsed[i] for i in pen_idx]
        lambda_pen  = ch.compute_tight_lambda(Q, parsed, pen_idx)
        slack_infos, n_slack = ch.determine_slack_variables(pen_parsed, n_x)
        slack_wires = list(range(n_x, n_x + n_slack))
        H_pen = base.build_penalty_hamiltonian(pen_parsed, slack_infos, lambda_pen, x_wires[0])
        H_prob = H_qubo + H_pen
    else:
        H_prob = H_qubo

    all_wires = x_wires + slack_wires
    num_gamma = base.count_gamma_terms(H_prob)

    dev = qml.device('lightning.qubit', wires=all_wires)

    @qml.qnode(dev)
    def circuit(g, b):
        for prep in state_prep:
            prep.opt_circuit()
        for w in slack_wires:
            qml.Hadamard(wires=w)
        gammas_2d = np.array([[g] * num_gamma])
        base.apply_cost_unitary(H_prob, gammas_2d, 0)
        base.apply_grover_mixer(b, all_wires, state_prep)
        return qml.expval(H_prob)

    return circuit


pu.setup_style()
C = pu._ROSE_PINE

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

gammas = np.linspace(-GAMMA_MAX, GAMMA_MAX, N_PTS)
betas  = np.linspace(-BETA_MAX,  BETA_MAX,  N_PTS)

summary_rows = []

for idx, task in enumerate(FOCUS_CASES):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    n_x = task['n_x']
    Q   = qubos[n_x][task['qubo_idx']]['Q']
    print(f"\nCase {task['case_num']}:", flush=True)

    circ = build_circuit_fn(task['constraints'], n_x, Q)

    # Coarse 2-D scan to find optimal (γ*, β*) and measure ranges
    N_COARSE = 20
    gc = np.linspace(-GAMMA_MAX, GAMMA_MAX, N_COARSE)
    bc = np.linspace(-BETA_MAX,  BETA_MAX,  N_COARSE)
    E2d = np.zeros((N_COARSE, N_COARSE))
    for ii, gg in enumerate(gc):
        for jj, bb in enumerate(bc):
            E2d[ii, jj] = float(circ(gg, bb))
    ij_min = np.unravel_index(E2d.argmin(), E2d.shape)
    ij_max = np.unravel_index(E2d.argmax(), E2d.shape)
    best_gamma = gc[ij_min[0]]
    best_beta  = bc[ij_min[1]]
    dE_total   = E2d.max() - E2d.min()
    print(f"  2D scan: best (γ,β) = ({best_gamma:.2f}, {best_beta:.2f})", flush=True)

    # β-slice at best-γ  (full resolution)
    E_beta_bg = np.array([float(circ(best_gamma, b)) for b in betas])
    # γ-slice at best-β  (full resolution)
    E_gamma_bb = np.array([float(circ(g, best_beta)) for g in gammas])

    dE_beta  = E_beta_bg.max()  - E_beta_bg.min()
    dE_gamma = E_gamma_bb.max() - E_gamma_bb.min()
    ratio    = dE_beta / dE_gamma if dE_gamma > 1e-6 else 0.0

    print(f"  ΔE total (2D)          = {dE_total:.3f}", flush=True)
    print(f"  ΔE_β  (at γ={best_gamma:.2f})  = {dE_beta:.3f}", flush=True)
    print(f"  ΔE_γ  (at β={best_beta:.2f})  = {dE_gamma:.3f}", flush=True)
    print(f"  ratio ΔE_β/ΔE_γ        = {ratio:.3f}", flush=True)

    summary_rows.append({
        'case': task['case_num'],
        'works': task['works'],
        'dE_total': dE_total,
        'dE_beta': dE_beta,
        'dE_gamma': dE_gamma,
        'ratio': ratio,
        'best_gamma': best_gamma,
        'best_beta': best_beta,
    })

    col_line = 'tab:green' if task['works'] else 'tab:red'
    ax2 = ax.twinx()
    ax.plot(betas, E_beta_bg, color=col_line, lw=2,
            label=f'E(β) @ γ={best_gamma:.2f}')
    ax2.plot(gammas, E_gamma_bb, color=col_line, lw=2, ls='--', alpha=0.7,
             label=f'E(γ) @ β={best_beta:.2f}')
    ax.set_xlabel('β  /  γ')
    ax.set_ylabel('⟨H⟩  (solid, β-axis)', color=col_line)
    ax2.set_ylabel('⟨H⟩  (dashed, γ-axis)', color=col_line, alpha=0.7)
    status = 'WORKS' if task['works'] else 'FLAT'
    ax.set_title(f'Case {task["case_num"]} [{status}]\n'
                 f'ΔE_β={dE_beta:.2f}  ΔE_γ={dE_gamma:.2f}  r={ratio:.2f}',
                 fontsize=8, color=col_line)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6)

fig.suptitle('p=1 Grover QAOA  –  β-slice at optimal γ\n'
             'Small ΔE_β / ΔE_γ ↔ mixer has no gradient  (flat landscape)',
             fontsize=12)

out = 'progress/figures/landscape_beta_slice.png'
pu.save_fig(fig, out)
print(f'\nSaved to {out}')

# Summary table
print('\n' + '='*72)
print(f'{"Case":>6}  {"Works":>6}  {"ΔE_tot":>8}  {"ΔE_β":>8}  {"ΔE_γ":>8}  {"β/γ":>6}')
print('-'*72)
for r in summary_rows:
    print(f'{r["case"]:>6}  {"Yes" if r["works"] else "No":>6}  '
          f'{r["dE_total"]:>8.3f}  {r["dE_beta"]:>8.3f}  {r["dE_gamma"]:>8.3f}  {r["ratio"]:>6.3f}')
