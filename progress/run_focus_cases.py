"""
run_focus_cases.py -- Run the 4 focus cases with VCG (flag) and VCGNoFlag.

Uses the exact same constraint instances as the server runs (from
run/params/experiment_params.jsonl), same QAOA budget, same layer sweep.
Writes results to progress/focus_results.pkl and prints a summary table.

Run from project root:
    python progress/run_focus_cases.py
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import pennylane as qml
from pennylane import numpy as np

from core import constraint_handler as ch
from core import hybrid_qaoa as hq
from core import dicke_state_prep as dsp
from core import qaoa_base as base
from core.vcg_no_flag import VCGNoFlag
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results.metrics import compute_comparison_metrics

os.makedirs('progress', exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
MAX_LAYERS        = 8
P_OPT_THRESHOLD   = 0.50   # stop layer sweep when P(opt) >= this
SHOTS             = 20_000
HYBRID_STEPS      = 150
HYBRID_RESTARTS   = 20
HYBRID_CQAOA_STEPS    = 50
HYBRID_CQAOA_RESTARTS = 10
HYBRID_LR         = 0.01

# VCGNoFlag gadget training budget
NF_QAOA_RESTARTS  = 10
NF_QAOA_STEPS     = 200
NF_MA_RESTARTS    = 20
NF_MA_STEPS       = 150
NF_LR             = 0.05
NF_THRESHOLD      = 0.999

# ── Focus cases (exact server instances) ──────────────────────────────────────
FOCUS_CASES = [
    {
        'name': 'independent_set+knapsack+cardinality',
        'constraints': ['x_0*x_1 == 0', '2*x_2 + 1*x_3 + 4*x_4 <= 2', 'x_5 + x_6 == 2'],
        'families': ['independent_set', 'knapsack', 'cardinality'],
        'n_x': 7, 'qubo_idx': 1,
    },
    {
        'name': 'knapsack+knapsack+cardinality',
        'constraints': ['3*x_0 + 2*x_1 <= 2', '2*x_2 + 1*x_3 <= 2', 'x_4 + x_5 + x_6 >= 3'],
        'families': ['knapsack', 'knapsack', 'cardinality'],
        'n_x': 7, 'qubo_idx': 8,
    },
    {
        'name': 'cardinality+knapsack',
        'constraints': ['x_0 + x_1 >= 2', '5*x_2 + 2*x_3 + 5*x_4 + 5*x_5 <= 9'],
        'families': ['cardinality', 'knapsack'],
        'n_x': 6, 'qubo_idx': 6,
    },
    {
        'name': 'cardinality+cardinality',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_3 + x_4 >= 1'],
        'families': ['cardinality', 'cardinality'],
        'n_x': 5, 'qubo_idx': 3,
    },
    # ── Cases with penalized constraints (overlapping variable sets) ───────────
    {
        'name': 'cardinality+cardinality(pen)+cardinality_geq',
        'constraints': ['x_0 + x_1 + x_2 == 1', 'x_0 + x_3 + x_4 == 1', 'x_5 + x_6 >= 1'],
        'families': ['cardinality', 'cardinality', 'cardinality_geq'],
        'n_x': 7, 'qubo_idx': 7,
        # partition_constraints: str=[0,2] (Dicke+VCG), pen=[1] (shares x_0)
    },
    {
        'name': 'knapsack(pen)+cardinality_leq+cardinality',
        'constraints': ['3*x_0 + 2*x_1 <= 2', 'x_1 + x_2 + x_3 <= 2', 'x_4 + x_5 + x_6 == 2'],
        'families': ['knapsack', 'cardinality_leq', 'cardinality'],
        'n_x': 7, 'qubo_idx': 1,
        # partition_constraints: str=[1,2] (LEQ+Dicke), pen=[0] (shares x_1)
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


def run_flag_variant(task, Q, parsed, str_idx, pen_idx,
                     delta_flag, lambda_pen, optimal_x, n_x):
    """Layer sweep with standard VCG (flag-based) gadgets."""
    rows = []
    prev_angles = None
    for p in range(1, MAX_LAYERS + 1):
        hybrid = hq.HybridQAOA(
            qubo=Q, all_constraints=parsed,
            structural_indices=str_idx, penalty_indices=pen_idx,
            penalty_str=[delta_flag], penalty_pen=lambda_pen,
            angle_strategy='ma-QAOA', mixer='Grover',
            n_layers=p, steps=HYBRID_STEPS, num_restarts=HYBRID_RESTARTS,
            learning_rate=HYBRID_LR,
            cqaoa_steps=HYBRID_CQAOA_STEPS,
            cqaoa_num_restarts=HYBRID_CQAOA_RESTARTS,
            gadget_db_path='gadgets/gadget_db.pkl',
        )
        opt_cost, counts, opt_angles = hybrid.solve()
        if prev_angles is not None:
            # warm-start: re-solve with prev as starting point
            pass  # HybridQAOA handles this via prev_layer_angles in optimize_angles
        m = get_metrics(counts, opt_cost, hybrid.problem_ham,
                        hybrid.all_wires, task['constraints'], n_x, optimal_x)
        rows.append({'p': p, **m})
        prev_angles = opt_angles
        print(f'    [flag]   p={p}: P(feas)={m["p_feasible"]:.3f}  P(opt)={m["p_optimal"]:.3f}', flush=True)
        if m['p_optimal'] >= P_OPT_THRESHOLD:
            break
    return rows


def run_noflag_variant(task, Q, parsed, str_idx, pen_idx,
                       lambda_pen, optimal_x, n_x):
    """Layer sweep with VCGNoFlag gadgets (manually assembled circuit)."""
    all_constraints = task['constraints']

    # Classify structural constraints
    dicke_idxs  = [i for i in str_idx if ch.is_dicke_compatible(parsed[i])]
    leq_idxs    = [i for i in str_idx if ch.is_cardinality_leq_compatible(parsed[i])]
    flow_idxs   = [i for i in str_idx if ch.is_flow_compatible(parsed[i])]
    gadget_idxs = [i for i in str_idx
                   if not ch.is_dicke_compatible(parsed[i])
                   and not ch.is_cardinality_leq_compatible(parsed[i])
                   and not ch.is_flow_compatible(parsed[i])]

    # Build structural preps
    dicke_preps = [dsp.from_parsed_constraint(parsed[i]) for i in dicke_idxs]
    leq_preps   = [dsp.from_cardinality_leq_constraint(parsed[i]) for i in leq_idxs]
    flow_preps  = [dsp.from_flow_constraint(parsed[i]) for i in flow_idxs]

    # Train VCGNoFlag for each gadget constraint (once, reused across layers)
    nf_gadgets = []
    for i in gadget_idxs:
        raw = parsed[i].raw
        print(f'    [noflag] Training VCGNoFlag: {raw}', flush=True)
        vcg_nf = VCGNoFlag(
            constraints=[raw],
            ar_threshold=NF_THRESHOLD, entropy_threshold=0.9,
            max_layers=8,
            qaoa_restarts=NF_QAOA_RESTARTS, qaoa_steps=NF_QAOA_STEPS,
            ma_restarts=NF_MA_RESTARTS, ma_steps=NF_MA_STEPS,
            lr=NF_LR, samples=SHOTS,
        )
        vcg_nf.train(verbose=False)
        ent_str = f'{vcg_nf.entropy:.4f}' if vcg_nf.entropy is not None else 'N/A'
        print(f'             AR={vcg_nf.ar:.4f}  entropy={ent_str}  '
              f'layers={vcg_nf.n_layers}  '
              f'n_feas={vcg_nf.n_feasible}  '
              f'single={vcg_nf._single_feasible_bitstring is not None}', flush=True)
        nf_gadgets.append(vcg_nf)

    state_prep = dicke_preps + leq_preps + flow_preps + nf_gadgets

    # Slack variables (for any penalty constraints)
    x_wires = list(range(n_x))
    pen_constraints = [parsed[i] for i in pen_idx]
    slack_offset = n_x  # no flag wires
    if pen_constraints:
        slack_infos, n_slack = ch.determine_slack_variables(pen_constraints, slack_offset)
        slack_wires = list(range(slack_offset, slack_offset + n_slack))
        H_penalty = base.build_penalty_hamiltonian(
            pen_constraints, slack_infos, lambda_pen, x_wires[0])
    else:
        slack_wires = []
        H_penalty = None

    all_wires = x_wires + slack_wires

    # Hamiltonian (QUBO + penalty, NO flag penalty)
    H_qubo = base.build_qubo_hamiltonian(Q, x_wires)
    H_prob = H_qubo + H_penalty if H_penalty is not None else H_qubo

    num_gamma = base.count_gamma_terms(H_prob)
    num_beta  = 1  # Grover mixer

    def evolution_circuit(angles, n_layers):
        @qml.qnode(qml.device('lightning.qubit', wires=all_wires))
        def circuit(angles):
            for prep in state_prep:
                prep.opt_circuit()
            for w in slack_wires:
                qml.Hadamard(wires=w)
            gammas, betas = base.split_angles(angles, num_gamma, num_beta, 'ma-QAOA')
            for q in range(n_layers):
                base.apply_cost_unitary(H_prob, gammas, q)
                base.apply_grover_mixer(betas[q][0], all_wires, state_prep)
            return qml.expval(H_prob)
        return circuit(angles)

    def counts_circuit(opt_angles, n_layers):
        @qml.qnode(qml.device('lightning.qubit', wires=all_wires, shots=SHOTS))
        def circuit():
            for prep in state_prep:
                prep.opt_circuit()
            for w in slack_wires:
                qml.Hadamard(wires=w)
            gammas, betas = base.split_angles(opt_angles, num_gamma, num_beta, 'ma-QAOA')
            for q in range(n_layers):
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
            steps=HYBRID_STEPS, num_restarts=HYBRID_RESTARTS,
            learning_rate=HYBRID_LR,
            prev_layer_angles=prev_angles,
        )
        counts = counts_circuit(opt_angles, p)
        m = get_metrics(counts, opt_cost, H_prob, all_wires,
                        all_constraints, n_x, optimal_x)
        rows.append({'p': p, **m})
        prev_angles = opt_angles
        print(f'    [noflag] p={p}: P(feas)={m["p_feasible"]:.3f}  P(opt)={m["p_optimal"]:.3f}', flush=True)
        if m['p_optimal'] >= P_OPT_THRESHOLD:
            break
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

all_results = {}

for task in FOCUS_CASES:
    name = task['name']
    n_x  = task['n_x']
    Q    = qubos[n_x][task['qubo_idx']]['Q']
    all_constraints = task['constraints']
    parsed = ch.parse_constraints(all_constraints)
    str_idx, pen_idx = ch.partition_constraints(parsed, strategy='auto')
    min_val, optimal_x, total_min = get_optimal_x(Q, all_constraints)
    # δ_flag: flag-qubit penalty for VCG(flag) gadgets — scaled to QUBO range
    delta_flag = float(5 + 2 * abs(total_min))
    # λ_pen: tight constraint penalty — minimum value guaranteeing all
    # penalized-constraint violations are costlier than f*
    lambda_pen = ch.compute_tight_lambda(Q, parsed, pen_idx)

    print('=' * 65, flush=True)
    print(f'Case: {name}  n={n_x}  opt={min_val:.3f}  '
          f'δ_flag={delta_flag:.1f}  λ_pen={lambda_pen:.1f}', flush=True)
    print(f'  Constraints: {all_constraints}', flush=True)
    print(f'  str_idx={str_idx}  pen_idx={pen_idx}', flush=True)
    print('=' * 65, flush=True)

    print('  >> VCG (flag):', flush=True)
    flag_rows = run_flag_variant(
        task, Q, parsed, str_idx, pen_idx, delta_flag, lambda_pen, optimal_x, n_x)

    print('  >> VCGNoFlag:', flush=True)
    nf_rows = run_noflag_variant(
        task, Q, parsed, str_idx, pen_idx, lambda_pen, optimal_x, n_x)

    all_results[name] = {'flag': flag_rows, 'noflag': nf_rows}
    print(flush=True)


# ── Print summary table ───────────────────────────────────────────────────────

print('\n' + '=' * 65, flush=True)
print('SUMMARY', flush=True)
print('=' * 65, flush=True)
for name, res in all_results.items():
    print(f'\n{name}:', flush=True)
    print(f'  {"Variant":<12} {"p":>3}  {"AR":>7}  {"P(feas)":>9}  {"P(opt)":>8}', flush=True)
    print('  ' + '-' * 45, flush=True)
    for row in res['flag']:
        print(f'  {"VCG(flag)":<12} {row["p"]:>3}  {row["AR"]:>7.4f}  {row["p_feasible"]:>9.4f}  {row["p_optimal"]:>8.4f}', flush=True)
    for row in res['noflag']:
        print(f'  {"VCGNoFlag":<12} {row["p"]:>3}  {row["AR"]:>7.4f}  {row["p_feasible"]:>9.4f}  {row["p_optimal"]:>8.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = 'progress/focus_results.pkl'
with open(out, 'wb') as f:
    pickle.dump(all_results, f)
print(f'\nSaved to {out}', flush=True)
