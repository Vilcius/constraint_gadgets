"""
run_hybrid_vs_penalty_jax.py -- Run HybridQAOACatalyst vs PenaltyQAOACatalyst.

Drop-in replacement for run_hybrid_vs_penalty.py that uses the JAX-jitted
solvers (default.qubit + jax.jit + jax.value_and_grad) instead of the
original PennyLane-based ones.  Both solvers use the same gradient method
(jax.value_and_grad) for direct comparability.

Usage
-----
Single task (SLURM):
    python run/run_hybrid_vs_penalty_jax.py \\
        --params run/params/experiment_params.jsonl \\
        --task-id 42 \\
        --db gadgets/vcg_db.pkl \\
        --pending-dir results/jax_pending_overlapping/

Merge per-task results:
    python run/run_hybrid_vs_penalty_jax.py \\
        --merge --pending-dir results/jax_pending_overlapping/ \\
        --output results/overlapping/hybrid_vs_penalty_jax.pkl
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

import jax
import jax.numpy as jnp
import pandas as pd

from core import constraint_handler as ch
from core import hybrid_qaoa_catalyst as hq
from core import penalty_qaoa_catalyst as pq
from core.qaoa_base import ising_hamiltonian_extremes
from data.make_data import read_qubos_from_file, get_optimal_x
from analyze_results.metrics import p_feasible_hybrid


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (mirrors run_hybrid_vs_penalty.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

MAX_LAYERS         = 5
P_FEAS_THRESHOLD   = 0.75
SHOTS              = 10_000

HYBRID_STEPS          = 50
HYBRID_RESTARTS       = 10
HYBRID_CQAOA_STEPS    = 30
HYBRID_CQAOA_RESTARTS = 10
HYBRID_LR             = 0.01

PENALTY_STEPS    = 50
PENALTY_RESTARTS = 10
PENALTY_LR       = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_rows(rows: list, method: str) -> list:
    out = []
    for row in rows:
        r = dict(row)
        r['method'] = [method]
        out.append(r)
    return out


def _save_rows(rows: list, path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            existing = pd.read_pickle(path)
            df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        except Exception:
            df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows)
    df.to_pickle(path)


# ─────────────────────────────────────────────────────────────────────────────
# Single-task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task: dict, qubos: dict, gadget_db_path: str,
             verbose: bool = True,
             hybrid_checkpoint_path: str = None) -> dict:
    """Run one HybridQAOACatalyst + PenaltyQAOACatalyst experiment."""
    all_constraints = task['constraints']
    n_x             = task['n_x']
    qubo_idx        = task['qubo_idx']

    Q_dict      = qubos[n_x][qubo_idx]
    Q           = Q_dict['Q']
    qubo_string = Q_dict['qubo_string']

    min_val, optimal_x, total_min = get_optimal_x(Q, all_constraints)
    penalty_weight = float(5 + 2 * abs(total_min))

    parsed = ch.parse_constraints(all_constraints)
    structural_indices, penalty_indices = ch.partition_constraints(parsed, strategy='auto')
    structural_constraints = [all_constraints[i] for i in structural_indices]
    penalty_constraints    = [all_constraints[i] for i in penalty_indices]

    if verbose:
        print(f'  QUBO({n_x}x{n_x})  opt={min_val:.3f}  δ={penalty_weight:.2f}')
        print(f'  Structural : {structural_constraints}')
        print(f'  Penalty    : {penalty_constraints}')

    # ── HybridQAOACatalyst layer sweep ───────────────────────────────────────
    hybrid_rows = []
    prev_h_angles = None
    h_cumulative_time = 0.0

    for p in range(1, MAX_LAYERS + 1):
        hybrid = hq.HybridQAOACatalyst(
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

        opt_cost, opt_angles = hybrid.optimize_angles(
            prev_layer_angles=prev_h_angles,
        )
        counts = hybrid.do_counts_circuit(shots=SHOTS)
        _, est_shots, est_error, grp_shots, grp_error = hybrid.get_circuit_resources()
        C_min, C_max = ising_hamiltonian_extremes(hybrid.problem_ham, hybrid.all_wires)

        row = {
            'constraint_type': ['+'.join(task.get('families', ['']))],
            'qubo_string':     [qubo_string],
            'constraints':     [all_constraints],
            'n_x':             [hybrid.n_x],
            'n_c':             [len(all_constraints)],
            'Hamiltonian':     [hybrid.problem_ham],
            'angle_strategy':  [hybrid.angle_strategy],
            'mixer':           [hybrid.mixer],
            'n_layers':        [hybrid.n_layers],
            'num_gamma':       [hybrid.num_gamma],
            'num_beta':        [hybrid.num_beta],
            'opt_angles':      [jnp.array(opt_angles).tolist()],
            'opt_cost':        [float(opt_cost)],
            'counts':          [counts],
            'est_shots':       [est_shots],
            'est_error':       [est_error],
            'group_est_shots': [grp_shots],
            'group_est_error': [grp_error],
            'hamiltonian_time':[getattr(hybrid, 'hamiltonian_time', None)],
            'optimize_time':   [hybrid.optimize_time],
            'counts_time':     [hybrid.count_time],
            'total_time':      [(getattr(hybrid, 'hamiltonian_time', None) or 0.0)
                                + hybrid.optimize_time + hybrid.count_time],
            'C_max':           [C_max],
            'C_min':           [C_min],
            'min_val':         [min_val],
            'AR':              [(float(opt_cost) - C_max) / (C_min - C_max)],
            'task':            [task],
            'layer':           [p],
            'optimal_x':       [optimal_x],
            'cumulative_time': [h_cumulative_time],   # filled below
        }
        h_cumulative_time += row['total_time'][0]
        row['cumulative_time'] = [h_cumulative_time]
        hybrid_rows.append(row)
        prev_h_angles = jnp.array(opt_angles)

        p_feas = p_feasible_hybrid(row)
        if verbose:
            ar = row['AR'][0]
            print(f'  HybridQAOACatalyst p={p}: AR={ar:.4f}  P(feas)={p_feas:.4f}')

        if p_feas >= P_FEAS_THRESHOLD:
            if verbose:
                print(f'    ✓ P(feasible) threshold reached at p={p}')
            break

    if hybrid_checkpoint_path is not None and hybrid_rows:
        _save_rows(_flatten_rows(hybrid_rows, 'HybridQAOACatalyst'), hybrid_checkpoint_path)
        print(f'  [checkpoint] HybridQAOACatalyst rows saved → {hybrid_checkpoint_path}',
              flush=True)

    # ── PenaltyQAOACatalyst layer sweep ──────────────────────────────────────
    penalty_rows = []
    prev_p_angles = None
    p_cumulative_time = 0.0

    for p in range(1, MAX_LAYERS + 1):
        pen_solver = pq.PenaltyQAOACatalyst(
            qubo=Q,
            constraints=all_constraints,
            penalty=penalty_weight,
            angle_strategy='ma-QAOA',
            n_layers=p,
            steps=PENALTY_STEPS,
            num_restarts=PENALTY_RESTARTS,
            learning_rate=PENALTY_LR,
        )

        C_min, C_max = ising_hamiltonian_extremes(pen_solver.full_Ham, pen_solver.all_wires)
        opt_cost, opt_angles = pen_solver.optimize_angles(
            prev_layer_angles=prev_p_angles,
        )
        _, est_shots, est_error, grp_shots, grp_error = pen_solver.get_circuit_resources()
        counts = pen_solver.do_counts_circuit(shots=SHOTS)

        row = {
            'constraint_type': ['+'.join(task.get('families', ['']))],
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
            'opt_angles':      [jnp.array(opt_angles).tolist()],
            'opt_cost':        [float(opt_cost)],
            'counts':          [counts],
            'est_shots':       [est_shots],
            'est_error':       [est_error],
            'group_est_shots': [grp_shots],
            'group_est_error': [grp_error],
            'optimize_time':   [pen_solver.optimize_time],
            'counts_time':     [pen_solver.count_time],
            'total_time':      [pen_solver.optimize_time + pen_solver.count_time],
            'cumulative_time': [p_cumulative_time + pen_solver.optimize_time
                                + pen_solver.count_time],
            'C_max':           [C_max],
            'C_min':           [C_min],
            'min_val':         [min_val],
            'AR':              [(float(opt_cost) - C_max) / (C_min - C_max)],
            'task':            [task],
            'layer':           [p],
            'optimal_x':       [optimal_x],
        }
        p_cumulative_time += pen_solver.optimize_time + pen_solver.count_time
        penalty_rows.append(row)
        prev_p_angles = jnp.array(opt_angles)

        p_feas = p_feasible_hybrid(row)
        if verbose:
            ar = row['AR'][0]
            print(f'  PenaltyQAOACatalyst p={p}: AR={ar:.4f}  P(feas)={p_feas:.4f}')

        if p_feas >= P_FEAS_THRESHOLD:
            if verbose:
                print(f'    ✓ P(feasible) threshold reached at p={p}')
            break

    return {'hybrid_rows': hybrid_rows, 'penalty_rows': penalty_rows}


# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def _merge(pending_dir: str, output: str, data_dir: str = 'data/') -> None:
    from analyze_results.results_helper import ResultsCollector
    from analyze_results.build_problem_table import build_problem_table_from_raw

    pattern = os.path.join(pending_dir, 'task_*.pkl')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No pending files found in {pending_dir}')
        return

    def _read_pkl_compat(path):
        """Read a task pkl, converting StringDtype columns to object dtype."""
        import pickle, io
        with open(path, 'rb') as f:
            raw = f.read()
        try:
            df = pd.read_pickle(io.BytesIO(raw))
        except Exception:
            # Pandas version mismatch: StringDtype(na_value=nan) not supported.
            # Patch by replacing the problematic dtype string before unpickling.
            import re
            # Replace StringDtype repr in raw bytes then re-read
            raw2 = raw.replace(b'StringDtype(storage=\'python\', na_value=nan)',
                               b'StringDtype()')
            try:
                df = pd.read_pickle(io.BytesIO(raw2))
            except Exception:
                df = pickle.loads(raw2)
        # Normalise any remaining StringDtype columns to object
        for col in df.columns:
            if hasattr(df[col], 'dtype') and hasattr(df[col].dtype, 'name') and 'string' in str(df[col].dtype).lower():
                df[col] = df[col].astype(object)
        return df

    collector = ResultsCollector()
    for fpath in files:
        try:
            df = _read_pkl_compat(fpath)
            for _, row in df.iterrows():
                collector.add(row.to_dict())
        except Exception as e:
            print(f'  [warn] Could not read {fpath}: {e}')

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    collector.save(output)
    n_rows = len(collector.to_dataframe())
    print(f'Merged {len(files)} task files → {output} ({n_rows} rows)')

    # Build problem_table automatically from the merged raw pkl
    output_dir  = os.path.dirname(output) or '.'
    output_stem = os.path.splitext(os.path.basename(output))[0]  # e.g. hybrid_vs_penalty_jax
    table_prefix = os.path.join(output_dir, output_stem.replace('hybrid_vs_penalty', 'problem_table'))
    print(f'\nBuilding problem_table → {table_prefix}.csv / .pkl')
    try:
        build_problem_table_from_raw(output, data_dir=data_dir, output_prefix=table_prefix)
    except Exception as e:
        import traceback
        print(f'  [warn] build_problem_table_from_raw failed: {e}')
        print(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Run HybridQAOACatalyst vs PenaltyQAOACatalyst layer sweeps.'
    )
    p.add_argument('--params', default='run/params/experiment_params.jsonl')
    p.add_argument('--task-id', type=int, default=None)
    p.add_argument('--db', default='gadgets/vcg_db.pkl')
    p.add_argument('--data-dir', default='data/')
    p.add_argument('--pending-dir', default='results/jax_pending_overlapping/')
    p.add_argument('--output', default='results/overlapping/hybrid_vs_penalty_jax.pkl')
    p.add_argument('--merge', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    args = _parse_args()

    if args.merge:
        _merge(args.pending_dir, args.output, data_dir=args.data_dir)
        sys.exit(0)

    if not os.path.exists(args.params):
        print(f'Params file not found: {args.params}')
        sys.exit(1)

    with open(args.params) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    qubos = read_qubos_from_file('qubos.csv', results_dir=args.data_dir)

    # ── Single task (SLURM) ──────────────────────────────────────────────────
    if args.task_id is not None:
        if args.task_id >= len(tasks):
            print(f'task-id {args.task_id} out of range ({len(tasks)} tasks). Nothing to do.')
            sys.exit(0)
        task = tasks[args.task_id]
        os.makedirs(args.pending_dir, exist_ok=True)
        result_path = os.path.join(args.pending_dir, f'task_{args.task_id}.pkl')

        print(f'Task {args.task_id}/{len(tasks)-1}')
        failure_out = os.path.join(args.pending_dir, f'task_{args.task_id}.failed.json')
        try:
            result = run_task(task, qubos, args.db,
                              hybrid_checkpoint_path=result_path)
        except Exception as e:
            tb = traceback.format_exc()
            print(f'ERROR: {e}\n{tb}', flush=True)
            with open(failure_out, 'w') as f:
                f.write(json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'task_id': args.task_id,
                    'task': task,
                    'error': str(e),
                    'traceback': tb,
                }) + '\n')
            sys.exit(1)

        penalty_rows = _flatten_rows(result['penalty_rows'], 'PenaltyQAOACatalyst')
        if penalty_rows:
            _save_rows(penalty_rows, result_path)
        n_total = len(pd.read_pickle(result_path)) if os.path.exists(result_path) else 0
        print(f'Saved {n_total} total rows to {result_path}')
        sys.exit(0)

    # ── Sequential mode ──────────────────────────────────────────────────────
    from analyze_results.results_helper import ResultsCollector
    print(f'Running {len(tasks)} tasks sequentially.')
    collector = ResultsCollector()

    for i, task in enumerate(tasks):
        print(f'\n[{i+1}/{len(tasks)}]')
        result = run_task(task, qubos, args.db)
        for row in result['hybrid_rows']:
            r = dict(row); r['method'] = ['HybridQAOACatalyst']; collector.add(r)
        for row in result['penalty_rows']:
            r = dict(row); r['method'] = ['PenaltyQAOACatalyst']; collector.add(r)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    collector.save(args.output)
    print(f'\nDone. Saved {len(collector.to_dataframe())} rows to {args.output}')
