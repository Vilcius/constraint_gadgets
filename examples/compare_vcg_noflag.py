"""
compare_vcg_noflag.py -- VCG (flag-based) vs VCGNoFlag inside HybridQAOA.

Runs the same 3-constraint problem as example_hybrid.py, but compares two
variants of HybridQAOA that differ only in how Constraint B is handled:

  Variant 1 (flag):    Constraint B → standard VCG (flag qubit at wire 7)
  Variant 2 (noflag):  Constraint B → VCGNoFlag   (no ancilla qubit)

Both variants:
  - Constraint A → Dicke state prep (exact, no flag qubit)
  - Constraint C → penalized in the cost Hamiltonian
  - Same QAOA hyperparameters (ma-QAOA, Grover mixer, p=1)

The VCGNoFlag variant is constructed manually using qaoa_base primitives
since HybridQAOA currently hardcodes VCG gadgets.  This comparison will
inform whether to replace VCG with VCGNoFlag in the codebase.

Run from project root:
    python examples/compare_vcg_noflag.py

Saves results to:
    examples/results/vcg_noflag_comparison.pkl
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
from analyze_results.metrics import compute_comparison_metrics, aggregate_counts

os.makedirs('examples/results', exist_ok=True)
os.makedirs('examples/figures', exist_ok=True)

# ─── Shared hyperparameters ───────────────────────────────────────────────────
N_X        = 7
SHOTS      = 10_000
N_LAYERS   = 1
STEPS      = 50
RESTARTS   = 10
LR         = 0.01

# VCG / VCGNoFlag gadget training budget
GADGET_STEPS    = 30
GADGET_RESTARTS = 10

# VCGNoFlag-specific (two-stage training)
NF_QAOA_RESTARTS = 5
NF_QAOA_STEPS    = 150
NF_MA_RESTARTS   = 10
NF_MA_STEPS      = 100
NF_LR            = 0.05
NF_THRESHOLD     = 0.999

# ─── Problem setup (identical to example_hybrid.py) ───────────────────────────
from analyze_results.results_helper import read_typed_csv, remap_constraint_to_vars

card_rows = read_typed_csv('data/cardinality_constraints.csv')
knap_rows = read_typed_csv('data/knapsack_constraints.csv')

c_a     = next(cs[0] for n, cs in card_rows if n == 3 and '== 1' in cs[0])
c_b_raw = next(cs[0] for n, cs in knap_rows if n == 3)
c_b     = remap_constraint_to_vars(c_b_raw, [3, 4, 5])
c_c_raw = next(cs[0] for n, cs in card_rows if n == 3 and '<= 1' in cs[0])
c_c     = remap_constraint_to_vars(c_c_raw, [1, 4, 6])

all_constraints = [c_a, c_b, c_c]
parsed          = ch.parse_constraints(all_constraints)

qubos           = read_qubos_from_file('qubos.csv', results_dir='data/')
Q               = qubos[N_X][0]['Q']
qubo_string     = qubos[N_X][0]['qubo_string']

min_val, optimal_x, total_min = get_optimal_x(Q, all_constraints)
penalty_weight  = float(5 + 2 * abs(total_min))

print("=" * 65)
print("VCG (flag) vs VCGNoFlag comparison — 3-constraint COP")
print("=" * 65)
print(f"  A (Dicke, structural): {c_a}")
print(f"  B (gadget):            {c_b}")
print(f"  C (penalized):         {c_c}")
print(f"  Optimal: {optimal_x}  value={min_val:.4f}  δ={penalty_weight:.2f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Variant 1 — HybridQAOA with standard VCG (flag-based) for Constraint B
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 65)
print("Variant 1: HybridQAOA with VCG (flag) for Constraint B")
print("─" * 65)

t0 = time.time()
hybrid_flag = hq.HybridQAOA(
    qubo              = Q,
    all_constraints   = parsed,
    structural_indices= [0, 1],
    penalty_indices   = [2],
    penalty_str       = [penalty_weight],
    penalty_pen       = penalty_weight,
    angle_strategy    = 'ma-QAOA',
    mixer             = 'Grover',
    n_layers          = N_LAYERS,
    steps             = STEPS,
    num_restarts      = RESTARTS,
    learning_rate     = LR,
    samples           = SHOTS,
    cqaoa_steps       = GADGET_STEPS,
    cqaoa_num_restarts= GADGET_RESTARTS,
)
opt_cost_flag, counts_flag, _ = hybrid_flag.solve()
t_flag = time.time() - t0

vcg_flag_gadget = hybrid_flag.gadget_preps[0]
vcg_flag_eigs   = qml.eigvals(vcg_flag_gadget.constraint_Ham)
vcg_flag_ar     = (float(vcg_flag_gadget.do_evolution_circuit(vcg_flag_gadget.opt_angles))
                   - float(max(vcg_flag_eigs))) / (
                   float(min(vcg_flag_eigs)) - float(max(vcg_flag_eigs)))

C_min_flag, C_max_flag = base.ising_hamiltonian_extremes(hybrid_flag.problem_ham, hybrid_flag.all_wires)
C_min_flag, C_max_flag = float(C_min_flag), float(C_max_flag)
m_flag = compute_comparison_metrics(
    counts_flag, opt_cost_flag, C_max_flag, C_min_flag,
    all_constraints, N_X, optimal_x,
)
print(f"  Gadget AR (Constraint B): {vcg_flag_ar:.4f}")
print(f"  Total time: {t_flag:.1f} s")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Variant 2 — HybridQAOA with VCGNoFlag for Constraint B
#
# Built manually using qaoa_base primitives since HybridQAOA hardcodes VCG.
# Wire layout: x_wires (0-6) + slack_wires (no flag wire for B)
# Hamiltonian: H_qubo + H_penalty_C   (no flag-penalty term for B)
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 65)
print("Variant 2: HybridQAOA with VCGNoFlag for Constraint B")
print("─" * 65)
t0 = time.time()

# ── 2a. Train VCGNoFlag for Constraint B ─────────────────────────────────────
print("  Training VCGNoFlag for Constraint B ...")
vcg_nf = VCGNoFlag(
    constraints    = [c_b],
    ar_threshold   = NF_THRESHOLD,
    max_layers     = 8,
    qaoa_restarts  = NF_QAOA_RESTARTS,
    qaoa_steps     = NF_QAOA_STEPS,
    ma_restarts    = NF_MA_RESTARTS,
    ma_steps       = NF_MA_STEPS,
    lr             = NF_LR,
    samples        = SHOTS,
)
vcg_nf.train(verbose=True)
print(f"  VCGNoFlag: AR={vcg_nf.ar:.4f}  layers={vcg_nf.n_layers}\n")

# ── 2b. Structural components: Dicke for A, VCGNoFlag for B ──────────────────
dicke_a = dsp.from_parsed_constraint(parsed[0])

# State prep list (Grover mixer needs this to uncompute/recompute)
nf_state_prep = [dicke_a, vcg_nf]

# ── 2c. Slack variables for penalized Constraint C ───────────────────────────
x_wires  = list(range(N_X))         # 0-6
pen_constraints = [parsed[2]]       # Constraint C
slack_offset = N_X                  # no flag wire → slack starts at 7

slack_infos, n_slack = ch.determine_slack_variables(pen_constraints, slack_offset)
slack_wires = list(range(slack_offset, slack_offset + n_slack))
all_wires_nf = x_wires + slack_wires

# ── 2d. Build Hamiltonian ─────────────────────────────────────────────────────
H_qubo_nf    = base.build_qubo_hamiltonian(Q, x_wires)
H_penalty_nf = base.build_penalty_hamiltonian(
    pen_constraints = pen_constraints,
    slack_infos     = slack_infos,
    delta           = penalty_weight,
    fallback_wire   = x_wires[0],
)
H_prob_nf = H_qubo_nf + H_penalty_nf   # no flag-penalty term

num_gamma_nf = base.count_gamma_terms(H_prob_nf)
num_beta_nf  = 1   # Grover mixer → 1 beta

# ── 2e. Optimise QAOA angles ──────────────────────────────────────────────────
def evolution_circuit_nf(angles):
    @qml.qnode(qml.device("lightning.qubit", wires=all_wires_nf))
    def circuit(angles):
        for prep in nf_state_prep:
            prep.opt_circuit()
        for wire in slack_wires:
            qml.Hadamard(wires=wire)
        gammas, betas = base.split_angles(angles, num_gamma_nf, num_beta_nf, 'ma-QAOA')
        for q in range(N_LAYERS):
            base.apply_cost_unitary(H_prob_nf, gammas, q)
            base.apply_grover_mixer(betas[q][0], all_wires_nf, nf_state_prep)
        return qml.expval(H_prob_nf)
    return circuit(angles)

print("  Optimising QAOA angles for Variant 2 ...")
opt_cost_nf, opt_angles_nf, opt_time_nf = base.run_optimization(
    cost_fn       = evolution_circuit_nf,
    n_layers      = N_LAYERS,
    num_gamma     = num_gamma_nf,
    num_beta      = num_beta_nf,
    angle_strategy= 'ma-QAOA',
    steps         = STEPS,
    num_restarts  = RESTARTS,
    learning_rate = LR,
)

# ── 2f. Sample counts ─────────────────────────────────────────────────────────
@qml.qnode(qml.device("lightning.qubit", wires=all_wires_nf, shots=SHOTS))
def counts_circuit_nf():
    for prep in nf_state_prep:
        prep.opt_circuit()
    for wire in slack_wires:
        qml.Hadamard(wires=wire)
    gammas, betas = base.split_angles(opt_angles_nf, num_gamma_nf, num_beta_nf, 'ma-QAOA')
    for q in range(N_LAYERS):
        base.apply_cost_unitary(H_prob_nf, gammas, q)
        base.apply_grover_mixer(betas[q][0], all_wires_nf, nf_state_prep)
    return qml.counts(all_outcomes=True)

counts_nf = counts_circuit_nf()
t_nf = time.time() - t0

C_min_nf, C_max_nf = base.ising_hamiltonian_extremes(H_prob_nf, all_wires_nf)
C_min_nf, C_max_nf = float(C_min_nf), float(C_max_nf)
m_nf = compute_comparison_metrics(
    counts_nf, opt_cost_nf, C_max_nf, C_min_nf,
    all_constraints, N_X, optimal_x,
)
print(f"  Total time: {t_nf:.1f} s")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("RESULTS")
print("=" * 65)
print()

# Gadget comparison (standalone)
print("Constraint B gadget (standalone):")
print(f"  {'Gadget':<16} {'AR':>8} {'P(feas)':>10} {'Qubits':>8} {'Layers':>8}")
print("  " + "-" * 54)
# VCG (flag) standalone
vcg_flag_counts = vcg_flag_gadget.do_counts_circuit(shots=SHOTS)
n_c_b = 1
vcg_flag_pfeas = sum(
    v for bs, v in vcg_flag_counts.items() if bs[-n_c_b:] == '0'
) / sum(vcg_flag_counts.values())
print(f"  {'VCG (flag)':<16} {vcg_flag_ar:>8.4f} {vcg_flag_pfeas:>10.4f} {len(vcg_flag_gadget.all_wires):>8} {vcg_flag_gadget.n_layers:>8}")
# VCGNoFlag standalone
vcg_nf_pfeas = vcg_nf.p_feasible()
print(f"  {'VCGNoFlag':<16} {vcg_nf.ar:>8.4f} {vcg_nf_pfeas:>10.4f} {len(vcg_nf.all_wires):>8} {vcg_nf.n_layers:>8}")
print()

# Full solver comparison
print("Full solver (HybridQAOA, p=1, ma-QAOA, Grover):")
print(f"  {'Variant':<22} {'AR':>8} {'P(feas)':>10} {'P(opt)':>8} {'Qubits':>8} {'Time(s)':>9}")
print("  " + "-" * 70)
print(f"  {'VCG (flag)':<22} {m_flag['AR']:>8.4f} {m_flag['p_feasible']:>10.4f} {m_flag['p_optimal']:>8.4f} {len(hybrid_flag.all_wires):>8} {t_flag:>9.1f}")
print(f"  {'VCGNoFlag':<22} {m_nf['AR']:>8.4f} {m_nf['p_feasible']:>10.4f} {m_nf['p_optimal']:>8.4f} {len(all_wires_nf):>8} {t_nf:>9.1f}")
print()
print("Note: AR is not comparable across variants — C_max differs because")
print("the flag variant includes flag-penalty terms in the Hamiltonian.")
print("Use P(feasible) and P(optimal) as the fair comparison metrics.")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    'problem': {
        'c_a': c_a, 'c_b': c_b, 'c_c': c_c,
        'qubo_string': qubo_string, 'optimal_x': optimal_x,
        'min_val': min_val, 'penalty_weight': penalty_weight,
    },
    'gadget_flag': {
        'ar': vcg_flag_ar, 'p_feasible': vcg_flag_pfeas,
        'n_layers': vcg_flag_gadget.n_layers,
        'n_qubits': len(vcg_flag_gadget.all_wires),
    },
    'gadget_noflag': {
        'ar': vcg_nf.ar, 'p_feasible': vcg_nf_pfeas,
        'n_layers': vcg_nf.n_layers,
        'n_qubits': len(vcg_nf.all_wires),
    },
    'solver_flag': {
        **m_flag,
        'n_qubits': len(hybrid_flag.all_wires),
        'time_s': t_flag,
        'counts': counts_flag,
    },
    'solver_noflag': {
        **m_nf,
        'n_qubits': len(all_wires_nf),
        'time_s': t_nf,
        'counts': counts_nf,
    },
}
out_path = 'examples/results/vcg_noflag_comparison.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to {out_path}")
