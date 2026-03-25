"""
plot_landscape.py -- Plot the p=1 QAOA energy landscape E(γ,β) for all 6 focus cases.

For each case we compute ⟨H_qubo⟩(γ,β) on a grid and plot the surface as a
heatmap.  Cases 1-3 work with Grover; Cases 4-6 fail (P(opt)=0).

Run from project root:
    python progress/plot_landscape.py
"""

import sys, os, itertools, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from core import constraint_handler as ch
from core import dicke_state_prep as dsp
from core import qaoa_base as base
from core.vcg import VCG
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results import plot_utils as pu

os.makedirs('progress/figures', exist_ok=True)

# ── Focus cases ──────────────────────────────────────────────────────────────
FOCUS_CASES = [
    {
        'case_num': 1, 'label': 'Case 1\nindep_set+knapsack+card',
        'constraints': ['x_0*x_1 == 0', '2*x_2 + 1*x_3 + 4*x_4 <= 2', 'x_5 + x_6 == 2'],
        'n_x': 7, 'qubo_idx': 1, 'works': True,
    },
    {
        'case_num': 2, 'label': 'Case 2\nknapsack+knapsack+card',
        'constraints': ['3*x_0 + 2*x_1 <= 2', '2*x_2 + 1*x_3 <= 2', 'x_4 + x_5 + x_6 >= 3'],
        'n_x': 7, 'qubo_idx': 8, 'works': True,
    },
    {
        'case_num': 3, 'label': 'Case 3\ncard+knapsack',
        'constraints': ['x_0 + x_1 >= 2', '5*x_2 + 2*x_3 + 5*x_4 + 5*x_5 <= 9'],
        'n_x': 6, 'qubo_idx': 6, 'works': True,
    },
    {
        'case_num': 4, 'label': 'Case 4\ncard+card  [FLAT]',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_3 + x_4 >= 1'],
        'n_x': 5, 'qubo_idx': 3, 'works': False,
    },
    {
        'case_num': 5, 'label': 'Case 5\ncard+card(pen)+card_geq  [FLAT]',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_0 + x_3 + x_4 == 1', 'x_5 + x_6 >= 1'],
        'n_x': 7, 'qubo_idx': 7, 'works': False,
    },
    {
        'case_num': 6, 'label': 'Case 6\nknapsack(pen)+card_leq+card  [FLAT]',
        'constraints': ['3*x_0 + 2*x_1 <= 2', 'x_1 + x_2 + x_3 <= 2', 'x_4 + x_5 + x_6 == 2'],
        'n_x': 7, 'qubo_idx': 1, 'works': False,
    },
]

# Grid parameters
N_GAMMA = 40
N_BETA  = 40
GAMMA_MAX = np.pi
BETA_MAX  = np.pi / 2

qubos = read_qubos_from_file('qubos.csv', results_dir='data/')

# NF training budget (light — we just need the state prep)
NF_THRESHOLD   = 0.999
NF_QAOA_REST   = 5
NF_QAOA_STEPS  = 100
NF_MA_REST     = 10
NF_MA_STEPS    = 80
NF_LR          = 0.05
NF_SHOTS       = 10_000


def build_state_preps(constraints, n_x):
    """Return (state_prep_list, all_wires, H_qubo, Q, optimal_x, min_val)."""
    parsed  = ch.parse_constraints(constraints)
    str_idx, pen_idx = ch.partition_constraints(parsed, strategy='auto')
    Q = None  # loaded separately

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
        print(f'    Training VCG: {raw}', flush=True)
        vcg_nf = VCG(
            constraints=[raw],
            ar_threshold=NF_THRESHOLD, max_layers=8,
            qaoa_restarts=NF_QAOA_REST, qaoa_steps=NF_QAOA_STEPS,
            ma_restarts=NF_MA_REST,     ma_steps=NF_MA_STEPS,
            lr=NF_LR, samples=NF_SHOTS,
        )
        vcg_nf.train(verbose=False)
        print(f'      AR={vcg_nf.ar:.4f}  layers={vcg_nf.n_layers}', flush=True)
        nf_gadgets.append(vcg_nf)

    state_prep = dicke_preps + leq_preps + flow_preps + nf_gadgets
    return state_prep, pen_idx, parsed


def compute_landscape(task, Q, optimal_x):
    """Return 2-D energy array E[i_gamma, i_beta]."""
    constraints = task['constraints']
    n_x = task['n_x']

    state_prep, pen_idx, parsed = build_state_preps(constraints, n_x)

    x_wires  = list(range(n_x))
    H_qubo   = base.build_qubo_hamiltonian(Q, x_wires)

    # Include penalty if needed
    slack_wires = []
    if pen_idx:
        pen_parsed = [parsed[i] for i in pen_idx]
        lambda_pen = ch.compute_tight_lambda(Q, parsed, pen_idx)
        slack_infos, n_slack = ch.determine_slack_variables(pen_parsed, n_x)
        slack_wires = list(range(n_x, n_x + n_slack))
        H_pen = base.build_penalty_hamiltonian(pen_parsed, slack_infos, lambda_pen, x_wires[0])
        H_prob = H_qubo + H_pen
    else:
        H_prob = H_qubo

    all_wires = x_wires + slack_wires
    num_gamma = base.count_gamma_terms(H_prob)
    num_beta  = 1

    gammas = np.linspace(-GAMMA_MAX, GAMMA_MAX, N_GAMMA)
    betas  = np.linspace(-BETA_MAX,  BETA_MAX,  N_BETA)

    E = np.zeros((N_GAMMA, N_BETA))

    dev = qml.device('lightning.qubit', wires=all_wires)

    @qml.qnode(dev)
    def circuit(g, b):
        for prep in state_prep:
            prep.opt_circuit()
        for w in slack_wires:
            qml.Hadamard(wires=w)
        # p=1: one cost + one mixer layer
        # apply_cost_unitary expects gammas shape (n_layers, num_gamma)
        gammas_2d = np.array([[g] * num_gamma])
        base.apply_cost_unitary(H_prob, gammas_2d, 0)
        base.apply_grover_mixer(b, all_wires, state_prep)
        return qml.expval(H_prob)

    total = N_GAMMA * N_BETA
    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            E[i, j] = float(circuit(g, b))
        if (i + 1) % 5 == 0:
            print(f'    {(i+1)*N_BETA}/{total} points done', flush=True)

    return E, gammas, betas


# ── Main ─────────────────────────────────────────────────────────────────────

pu.setup_style()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('p=1 QAOA energy landscape  ⟨H⟩(γ, β)\n'
             'Rows: working (top) vs flat-landscape (bottom)', fontsize=13)

for ax, task in zip(axes.flatten(), FOCUS_CASES):
    n_x   = task['n_x']
    Q     = qubos[n_x][task['qubo_idx']]['Q']
    min_val, optimal_x, _ = get_optimal_x(Q, task['constraints'])

    print(f"\nCase {task['case_num']}: {task['label'].replace(chr(10), ' ')}", flush=True)
    E, gammas, betas = compute_landscape(task, Q, optimal_x)

    # Normalise to [0,1]: we want to see the *shape*, not scale
    E_min, E_max = E.min(), E.max()
    E_range = E_max - E_min if E_max > E_min else 1.0
    E_norm = (E - E_min) / E_range

    im = ax.imshow(
        E_norm.T,
        origin='lower', aspect='auto',
        extent=[-GAMMA_MAX, GAMMA_MAX, -BETA_MAX, BETA_MAX],
        cmap='viridis',
        vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax, label='⟨H⟩ (normalised)')

    # Mark the minimum with a cross
    ij = np.unravel_index(E.argmin(), E.shape)
    ax.plot(gammas[ij[0]], betas[ij[1]], 'r+', markersize=12, markeredgewidth=2,
            label=f'min γ={gammas[ij[0]]:.2f} β={betas[ij[1]]:.2f}')

    ax.set_xlabel('γ')
    ax.set_ylabel('β')
    col = 'tab:green' if task['works'] else 'tab:red'
    ax.set_title(task['label'] + f'\nΔE={E_range:.2f}', fontsize=8, color=col)
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
out = 'progress/figures/qaoa_landscape_p1.png'
pu.save_fig(fig, out)
print(f'\nSaved to {out}')
