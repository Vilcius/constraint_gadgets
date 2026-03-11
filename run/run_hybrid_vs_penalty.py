"""
run_hybrid_vs_penalty.py -- Run HybridQAOA vs PenaltyQAOA layer sweeps.

For each experiment task:
  - HybridQAOA   : sweep p=1..MAX_LAYERS, stop when P(feasible) >= P_FEAS_THRESHOLD
  - PenaltyQAOA  : sweep p=1..MAX_LAYERS, stop when P(feasible) >= P_FEAS_THRESHOLD

Both solvers use ma-QAOA angles with warm-started layer growth (previous
optimal angles are passed as the starting point for the next layer).

P(feasible) is computed over ALL constraints (structural + penalty).

Modes
-----
Sequential (process all tasks in the params file):
    python run/run_hybrid_vs_penalty.py \\
        --params run/params/experiment_params.jsonl \\
        --db gadgets/gadget_db.pkl

Single task (SLURM):
    python run/run_hybrid_vs_penalty.py \\
        --params run/params/experiment_params.jsonl \\
        --task-id 42 \\
        --db gadgets/gadget_db.pkl \\
        --pending-dir results/pending/

Merge per-task results:
    python run/run_hybrid_vs_penalty.py \\
        --merge --pending-dir results/pending/ \\
        --output results/hybrid_vs_penalty.pkl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import json
import argparse
import glob
import traceback
from datetime import datetime

import pandas as pd
from pennylane import numpy as np

from core import constraint_handler as ch
from core import hybrid_qaoa as hq
from core import penalty_qaoa as pq
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results.results_helper import (
    ResultsCollector,
    collect_hybrid_data, collect_penalty_data,
)
from analyze_results.metrics import p_feasible_hybrid


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_LAYERS         = 5      # maximum QAOA layers for both methods
P_FEAS_THRESHOLD   = 0.75   # stop adding layers when P(feasible) >= this
SHOTS              = 10_000 # measurement shots for counts

# HybridQAOA optimisation budget
HYBRID_STEPS       = 50
HYBRID_RESTARTS    = 10
HYBRID_CQAOA_STEPS = 30
HYBRID_CQAOA_RESTARTS = 10
HYBRID_LR          = 0.01

# PenaltyQAOA optimisation budget
PENALTY_STEPS      = 50
PENALTY_RESTARTS   = 10
PENALTY_LR         = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Single-task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task: dict, qubos: dict, gadget_db_path: str,
             verbose: bool = True) -> dict:
    """Run one HybridQAOA + PenaltyQAOA experiment.

    Parameters
    ----------
    task : dict
        Task dict as produced by generate_experiment_params.py.
    qubos : dict
        Pre-loaded QUBO dictionary (from read_qubos_from_file).
    gadget_db_path : str
        Path to the pre-built GadgetDatabase pickle.

    Returns
    -------
    dict with keys 'hybrid_rows' and 'penalty_rows', each a list of
    single-row result dicts (one per QAOA layer).
    """
    structural_constraints = task['structural_constraints']
    penalty_constraints    = task['penalty_constraints']
    all_constraints        = structural_constraints + penalty_constraints
    structural_indices     = task['structural_indices']
    penalty_indices        = task['penalty_indices']
    n_x                    = task['n_x']
    qubo_idx               = task['qubo_idx']

    Q_dict      = qubos[n_x][qubo_idx]
    Q           = Q_dict['Q']
    qubo_string = Q_dict['qubo_string']

    # Brute-force optimal and penalty weight
    min_val, optimal_x, total_min = get_optimal_x(Q, all_constraints)
    penalty_weight = float(5 + 2 * abs(total_min))

    parsed = ch.parse_constraints(all_constraints)

    if verbose:
        print(f'  QUBO({n_x}x{n_x})  opt={min_val:.3f}  δ={penalty_weight:.2f}')
        print(f'  Structural : {structural_constraints}')
        print(f'  Penalty    : {penalty_constraints}')

    # ── HybridQAOA layer sweep ───────────────────────────────────────────────
    hybrid_rows = []
    prev_h_angles = None

    for p in range(1, MAX_LAYERS + 1):
        hybrid = hq.HybridQAOA(
            qubo=Q,
            all_constraints=parsed,
            structural_indices=structural_indices,
            penalty_indices=penalty_indices,
            penalty_str=[penalty_weight],
            penalty_pen=penalty_weight,
            angle_strategy='ma-QAOA',
            mixer='Grover',
            n_layers=p,
            steps=HYBRID_STEPS,
            num_restarts=HYBRID_RESTARTS,
            learning_rate=HYBRID_LR,
            cqaoa_steps=HYBRID_CQAOA_STEPS,
            cqaoa_num_restarts=HYBRID_CQAOA_RESTARTS,
            gadget_db_path=gadget_db_path,
        )

        row = collect_hybrid_data(
            all_constraints, hybrid, qubo_string,
            min_val=min_val,
            previous_angles=prev_h_angles,
            constraint_type=task.get('structural_families', [''])[0],
        )
        row['task']       = [task]
        row['layer']      = [p]
        row['optimal_x']  = [optimal_x]
        hybrid_rows.append(row)
        prev_h_angles = np.array(row['opt_angles'][0])

        p_feas = p_feasible_hybrid(row)
        if verbose:
            ar = row['AR'][0]
            print(f'  HybridQAOA p={p}: AR={ar:.4f}  P(feas)={p_feas:.4f}')

        if p_feas >= P_FEAS_THRESHOLD:
            if verbose:
                print(f'    ✓ P(feasible) threshold reached at p={p}')
            break

    # ── PenaltyQAOA layer sweep ──────────────────────────────────────────────
    penalty_rows = []
    prev_p_angles = None

    for p in range(1, MAX_LAYERS + 1):
        pen_solver = pq.PenaltyQAOA(
            qubo=Q,
            constraints=all_constraints,
            penalty=penalty_weight,
            angle_strategy='ma-QAOA',
            n_layers=p,
            steps=PENALTY_STEPS,
            num_restarts=PENALTY_RESTARTS,
            learning_rate=PENALTY_LR,
        )

        # Optimise with warm-start from previous layer
        import pennylane as qml
        C_max = float(max(qml.eigvals(pen_solver.full_Ham)))
        C_min = float(min(qml.eigvals(pen_solver.full_Ham)))
        opt_cost, opt_angles = pen_solver.optimize_angles(
            pen_solver.do_evolution_circuit,
            prev_layer_angles=prev_p_angles,
        )
        _, est_shots, est_error, group_est_shots, group_est_error = pen_solver.get_circuit_resources()
        counts = pen_solver.do_counts_circuit(shots=SHOTS)

        row = {
            'constraint_type': [task.get('structural_families', [''])[0]],
            'qubo_string':     [qubo_string],
            'constraints':     [all_constraints],
            'n_x':             [pen_solver.n_x],
            'n_c':             [len(all_constraints)],
            'Hamiltonian':     [pen_solver.full_Ham],
            'angle_strategy':  [pen_solver.angle_strategy],
            'penalty':         [pen_solver.penalty_param],
            'n_layers':        [pen_solver.n_layers],
            'num_gamma':       [pen_solver.num_gamma],
            'num_beta':        [pen_solver.num_beta],
            'opt_angles':      [opt_angles.tolist()],
            'opt_cost':        [float(opt_cost)],
            'counts':          [counts],
            'est_shots':       [est_shots],
            'est_error':       [est_error],
            'group_est_shots': [group_est_shots],
            'group_est_error': [group_est_error],
            'optimize_time':   [pen_solver.optimize_time],
            'counts_time':     [pen_solver.count_time],
            'C_max':           [C_max],
            'C_min':           [C_min],
            'min_val':         [min_val],
            'AR':              [(float(opt_cost) - C_max) / (C_min - C_max)],
            'task':            [task],
            'layer':           [p],
            'optimal_x':       [optimal_x],
        }
        penalty_rows.append(row)
        prev_p_angles = np.array(opt_angles)

        p_feas = p_feasible_hybrid(row)
        if verbose:
            ar = row['AR'][0]
            print(f'  PenaltyQAOA p={p}: AR={ar:.4f}  P(feas)={p_feas:.4f}')

        if p_feas >= P_FEAS_THRESHOLD:
            if verbose:
                print(f'    ✓ P(feasible) threshold reached at p={p}')
            break

    return {'hybrid_rows': hybrid_rows, 'penalty_rows': penalty_rows}


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def _merge(pending_dir: str, output: str) -> None:
    """Combine per-task pending pickles into one results file."""
    pattern = os.path.join(pending_dir, 'task_*.pkl')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No pending files found in {pending_dir}')
        return

    collector = ResultsCollector()
    for fpath in files:
        try:
            df = pd.read_pickle(fpath)
            for _, row in df.iterrows():
                collector.add(row.to_dict())
        except Exception as e:
            print(f'  [warn] Could not read {fpath}: {e}')

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    collector.save(output)
    print(f'Merged {len(files)} task files → {output} ({len(collector.to_dataframe())} rows)')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Run HybridQAOA vs PenaltyQAOA layer sweeps.'
    )
    p.add_argument('--params', default='run/params/experiment_params.jsonl',
                   help='JSON-lines params file from generate_experiment_params.py')
    p.add_argument('--task-id', type=int, default=None,
                   help='Process only the N-th task (SLURM mode).')
    p.add_argument('--db', default='gadgets/gadget_db.pkl',
                   help='Path to the pre-built GadgetDatabase pickle.')
    p.add_argument('--data-dir', default='data/')
    p.add_argument('--pending-dir', default='results/pending/',
                   help='Directory for per-task result pickles.')
    p.add_argument('--output', default='results/hybrid_vs_penalty.pkl',
                   help='Final merged results pickle.')
    p.add_argument('--merge', action='store_true',
                   help='Merge pending per-task pickles into --output.')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # ── Merge mode ───────────────────────────────────────────────────────────
    if args.merge:
        _merge(args.pending_dir, args.output)
        sys.exit(0)

    # ── Load shared data ─────────────────────────────────────────────────────
    if not os.path.exists(args.params):
        print(f'Params file not found: {args.params}')
        print('Run generate_experiment_params.py first.')
        sys.exit(1)

    with open(args.params) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    qubos = read_qubos_from_file('qubos.csv', results_dir=args.data_dir)

    # ── Single task (SLURM) ──────────────────────────────────────────────────
    if args.task_id is not None:
        if args.task_id >= len(tasks):
            print(f'task-id {args.task_id} out of range (have {len(tasks)} tasks). Nothing to do.')
            sys.exit(0)
        task = tasks[args.task_id]
        os.makedirs(args.pending_dir, exist_ok=True)
        result_path = os.path.join(args.pending_dir, f'task_{args.task_id}.pkl')

        print(f'Task {args.task_id}/{len(tasks)-1}')
        failure_out = os.path.join(args.pending_dir, f'task_{args.task_id}.failed.json')
        try:
            result = run_task(task, qubos, args.db)
        except Exception as e:
            tb = traceback.format_exc()
            print(f'ERROR: {e}\n{tb}', flush=True)
            failure = {
                'timestamp': datetime.now().isoformat(),
                'task_id': args.task_id,
                'task': task,
                'error': str(e),
                'traceback': tb,
            }
            with open(failure_out, 'w') as f:
                f.write(json.dumps(failure) + '\n')
            sys.exit(1)

        # Flatten all rows into one DataFrame
        all_rows = []
        for row in result['hybrid_rows']:
            r = dict(row)
            r['method'] = ['HybridQAOA']
            all_rows.append(r)
        for row in result['penalty_rows']:
            r = dict(row)
            r['method'] = ['PenaltyQAOA']
            all_rows.append(r)

        if all_rows:
            pd.DataFrame(all_rows).to_pickle(result_path)
            print(f'Saved {len(all_rows)} rows to {result_path}')
        sys.exit(0)

    # ── Sequential mode ──────────────────────────────────────────────────────
    print(f'Running {len(tasks)} tasks sequentially.')
    collector = ResultsCollector()

    for i, task in enumerate(tasks):
        print(f'\n[{i+1}/{len(tasks)}]')
        result = run_task(task, qubos, args.db)
        for row in result['hybrid_rows']:
            r = dict(row); r['method'] = ['HybridQAOA']; collector.add(r)
        for row in result['penalty_rows']:
            r = dict(row); r['method'] = ['PenaltyQAOA']; collector.add(r)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    collector.save(args.output)
    print(f'\nDone. Saved {len(collector.to_dataframe())} rows to {args.output}')
