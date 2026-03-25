"""
run_xy_mixer_test.py  --  Test XY mixer vs Grover on the three flat-landscape cases.

Grover flat-landscape root cause
---------------------------------
For Cases 4, 5, 6 the Grover mixer G = A(2|0><0|-I)A† is nearly useless because
A|0> is close to the uniform superposition over F.  When A†|psi_0> = |0>, the
mixer satisfies G|psi_0> = |psi_0>, i.e. no update.  The gradient d<H>/d_beta
at beta=0 vanishes for the uniform feasible state regardless of circuit depth.

Fix being tested
-----------------
Keep only exact Dicke (sum==k) constraints as structural — their XY mixer
exactly preserves the Hamming weight, so those constraints are hard-enforced.
Move all remaining structural constraints (VCG/LEQ types) to penalty with
tight lambda.  Use mixer="XY":
  - Ring-XY on Dicke wires   -> hard feasibility within each Dicke group
  - RX on all remaining wires -> non-zero gradient everywhere, no flat landscape

Case partition for this test
-----------------------------
Case 4: str=[x_0+x_1+x_2==1(Dicke)],   pen=[x_3+x_4>=1]
Case 5: str=[x_0+x_1+x_2==1(Dicke)],   pen=[x_0+x_3+x_4==1, x_5+x_6>=1]
Case 6: str=[x_4+x_5+x_6==2(Dicke)],   pen=[3x_0+2x_1<=2, x_1+x_2+x_3<=2]

Run from project root:
    python progress/run_xy_mixer_test.py
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import pennylane as qml
from pennylane import numpy as np

from core import constraint_handler as ch
from core import hybrid_qaoa as hq
from core import qaoa_base as base
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results.metrics import compute_comparison_metrics

os.makedirs('progress', exist_ok=True)

# ── Hyperparameters (match run_focus_cases.py) ─────────────────────────────────
MAX_LAYERS       = 8
P_OPT_THRESHOLD  = 0.50
SHOTS            = 20_000
STEPS            = 150
RESTARTS         = 20
LR               = 0.01

# ── Flat-landscape cases with explicit structural/penalty partition ─────────────
# For each case, str_idx and pen_idx are manually set to use Dicke-only structural.
FLAT_CASES = [
    {
        'case_num': 4,
        'name': 'cardinality+cardinality',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_3 + x_4 >= 1'],
        'n_x': 5, 'qubo_idx': 3,
        # Dicke-only structural; VCG constraint -> penalty
        'str_idx': [0],
        'pen_idx': [1],
    },
    {
        'case_num': 5,
        'name': 'cardinality+cardinality(pen)+cardinality_geq',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_0 + x_3 + x_4 == 1', 'x_5 + x_6 >= 1'],
        'n_x': 7, 'qubo_idx': 7,
        # Dicke-only structural; both others -> penalty
        'str_idx': [0],
        'pen_idx': [1, 2],
    },
    {
        'case_num': 6,
        'name': 'knapsack(pen)+cardinality_leq+cardinality',
        'constraints': ['3*x_0 + 2*x_1 <= 2', 'x_1 + x_2 + x_3 <= 2', 'x_4 + x_5 + x_6 == 2'],
        'n_x': 7, 'qubo_idx': 1,
        # Dicke-only structural; knapsack + LEQ -> penalty
        'str_idx': [2],
        'pen_idx': [0, 1],
    },
]

qubos = read_qubos_from_file('qubos.csv', results_dir='data/')


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_metrics(counts, opt_cost, ham, all_wires, constraints, n_x, optimal_x):
    C_min, C_max = base.ising_hamiltonian_extremes(ham, all_wires)
    return compute_comparison_metrics(
        counts, opt_cost, float(C_max), float(C_min),
        constraints, n_x, optimal_x,
    )


def run_grover_vcg(task, Q, parsed, str_idx, pen_idx, lambda_pen, optimal_x, n_x):
    """Grover + VCG baseline (original approach for these cases)."""
    from core.vcg import VCG
    import core.dicke_state_prep as dsp

    dicke_idxs  = [i for i in str_idx if ch.is_dicke_compatible(parsed[i])]
    leq_idxs    = [i for i in str_idx if ch.is_cardinality_leq_compatible(parsed[i])]
    gadget_idxs = [i for i in str_idx
                   if not ch.is_dicke_compatible(parsed[i])
                   and not ch.is_cardinality_leq_compatible(parsed[i])
                   and not ch.is_flow_compatible(parsed[i])]

    dicke_preps = [dsp.from_parsed_constraint(parsed[i]) for i in dicke_idxs]
    leq_preps   = [dsp.from_cardinality_leq_constraint(parsed[i]) for i in leq_idxs]

    nf_gadgets = []
    for i in gadget_idxs:
        raw = parsed[i].raw
        print(f'    [grover] Training VCG: {raw}', flush=True)
        vcg_nf = VCG(
            constraints=[raw],
            ar_threshold=0.999, entropy_threshold=0.9,
            max_layers=8,
            qaoa_restarts=10, qaoa_steps=200,
            ma_restarts=20, ma_steps=150,
            lr=0.05, samples=SHOTS,
        )
        vcg_nf.train(verbose=False)
        print(f'             AR={vcg_nf.ar:.4f}  entropy={vcg_nf.entropy:.4f if vcg_nf.entropy else "N/A"}  '
              f'layers={vcg_nf.n_layers}', flush=True)
        nf_gadgets.append(vcg_nf)

    state_prep = dicke_preps + leq_preps + nf_gadgets
    x_wires = list(range(n_x))
    pen_constraints = [parsed[i] for i in pen_idx]
    slack_offset = n_x
    if pen_constraints:
        slack_infos, n_slack = ch.determine_slack_variables(pen_constraints, slack_offset)
        slack_wires = list(range(slack_offset, slack_offset + n_slack))
        H_penalty = base.build_penalty_hamiltonian(pen_constraints, slack_infos, lambda_pen, x_wires[0])
    else:
        slack_wires = []
        H_penalty = None

    all_wires = x_wires + slack_wires
    H_qubo = base.build_qubo_hamiltonian(Q, x_wires)
    H_prob = H_qubo + H_penalty if H_penalty is not None else H_qubo
    num_gamma = base.count_gamma_terms(H_prob)
    num_beta  = 1  # Grover

    def evolution_circuit(angles, p):
        @qml.qnode(qml.device('lightning.qubit', wires=all_wires))
        def circuit(angles):
            for prep in state_prep:
                prep.opt_circuit()
            for w in slack_wires:
                qml.Hadamard(wires=w)
            gammas, betas = base.split_angles(angles, num_gamma, num_beta, 'ma-QAOA')
            for q in range(p):
                base.apply_cost_unitary(H_prob, gammas, q)
                base.apply_grover_mixer(betas[q][0], all_wires, state_prep)
            return qml.expval(H_prob)
        return circuit(angles)

    def counts_circuit(opt_angles, p):
        @qml.qnode(qml.device('lightning.qubit', wires=all_wires, shots=SHOTS))
        def circuit():
            for prep in state_prep:
                prep.opt_circuit()
            for w in slack_wires:
                qml.Hadamard(wires=w)
            gammas, betas = base.split_angles(opt_angles, num_gamma, num_beta, 'ma-QAOA')
            for q in range(p):
                base.apply_cost_unitary(H_prob, gammas, q)
                base.apply_grover_mixer(betas[q][0], all_wires, state_prep)
            return qml.counts(all_outcomes=True)
        return circuit()

    rows = []
    prev_angles = None
    for p in range(1, MAX_LAYERS + 1):
        opt_cost, opt_angles, _ = base.run_optimization(
            cost_fn=lambda a: evolution_circuit(a, p),
            n_layers=p, num_gamma=num_gamma, num_beta=num_beta,
            angle_strategy='ma-QAOA',
            steps=STEPS, num_restarts=RESTARTS, learning_rate=LR,
            prev_layer_angles=prev_angles,
        )
        counts = counts_circuit(opt_angles, p)
        m = get_metrics(counts, opt_cost, H_prob, all_wires,
                        task['constraints'], n_x, optimal_x)
        rows.append({'p': p, **m})
        prev_angles = opt_angles
        print(f'    [grover]  p={p}: P(feas)={m["p_feasible"]:.3f}  P(opt)={m["p_optimal"]:.3f}', flush=True)
        if m['p_optimal'] >= P_OPT_THRESHOLD:
            break
    return rows


def run_xy_mixer(task, Q, parsed, str_idx, pen_idx, lambda_pen, optimal_x, n_x):
    """
    XY mixer variant: Dicke constraints structural, rest as penalty.

    Mixer:
      - Ring-XY on each Dicke group (hard-preserves Hamming weight)
      - RX on all remaining wires (flag-free gradient signal, no flat landscape)
    """
    hybrid = hq.HybridQAOA(
        qubo=Q, all_constraints=parsed,
        structural_indices=str_idx, penalty_indices=pen_idx,
        penalty_pen=lambda_pen,
        angle_strategy='ma-QAOA',
        mixer='XY',           # Ring-XY on Dicke wires, RX on rest
        n_layers=1,           # placeholder — loop overrides below
        steps=STEPS, num_restarts=RESTARTS, learning_rate=LR,
        samples=SHOTS,
        cqaoa_steps=50, cqaoa_num_restarts=10,
    )

    rows = []
    prev_angles = None
    for p in range(1, MAX_LAYERS + 1):
        hybrid.n_layers = p
        opt_cost, opt_angles = hybrid.optimize_angles(
            hybrid.do_evolution_circuit,
            prev_layer_angles=prev_angles,
        )
        counts = hybrid.do_counts_circuit(angles=opt_angles, shots=SHOTS)
        m = get_metrics(counts, opt_cost, hybrid.problem_ham,
                        hybrid.all_wires, task['constraints'], n_x, optimal_x)
        rows.append({'p': p, **m})
        prev_angles = opt_angles
        print(f'    [xy]      p={p}: P(feas)={m["p_feasible"]:.3f}  P(opt)={m["p_optimal"]:.3f}', flush=True)
        if m['p_optimal'] >= P_OPT_THRESHOLD:
            break
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

all_results = {}

for task in FLAT_CASES:
    name  = task['name']
    n_x   = task['n_x']
    Q     = qubos[n_x][task['qubo_idx']]['Q']
    parsed = ch.parse_constraints(task['constraints'])
    str_idx = task['str_idx']
    pen_idx = task['pen_idx']

    min_val, optimal_x, total_min = get_optimal_x(Q, task['constraints'])
    lambda_pen = ch.compute_tight_lambda(Q, parsed, pen_idx)

    print('=' * 65, flush=True)
    print(f'Case {task["case_num"]}: {name}  n={n_x}  f*={min_val:.3f}  λ={lambda_pen:.1f}', flush=True)
    print(f'  Constraints: {task["constraints"]}', flush=True)
    print(f'  str_idx={str_idx}  pen_idx={pen_idx}', flush=True)
    print('=' * 65, flush=True)

    print('  >> Grover + VCG (baseline):', flush=True)
    grover_rows = run_grover_vcg(
        task, Q, parsed, str_idx, pen_idx, lambda_pen, optimal_x, n_x)

    print('  >> XY mixer (Dicke structural + penalty rest):', flush=True)
    xy_rows = run_xy_mixer(
        task, Q, parsed, str_idx, pen_idx, lambda_pen, optimal_x, n_x)

    all_results[name] = {'grover': grover_rows, 'xy': xy_rows}
    print(flush=True)


# ── Summary ───────────────────────────────────────────────────────────────────

print('\n' + '=' * 65, flush=True)
print('SUMMARY  (flat-landscape cases: Grover vs XY mixer)', flush=True)
print('=' * 65, flush=True)

for task in FLAT_CASES:
    name = task['name']
    res  = all_results[name]
    print(f'\nCase {task["case_num"]}: {name}', flush=True)
    print(f'  {"Variant":<16} {"p":>3}  {"P(feas)":>9}  {"P(opt)":>8}', flush=True)
    print('  ' + '-' * 42, flush=True)
    best_grover = max(res['grover'], key=lambda r: r['p_optimal'])
    best_xy     = max(res['xy'],     key=lambda r: r['p_optimal'])
    print(f'  {"Grover+VCG":<16} {best_grover["p"]:>3}  '
          f'{best_grover["p_feasible"]:>9.4f}  {best_grover["p_optimal"]:>8.4f}', flush=True)
    print(f'  {"XY mixer":<16} {best_xy["p"]:>3}  '
          f'{best_xy["p_feasible"]:>9.4f}  {best_xy["p_optimal"]:>8.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = 'progress/xy_mixer_results.pkl'
with open(out, 'wb') as f:
    pickle.dump(all_results, f)
print(f'\nSaved to {out}', flush=True)
