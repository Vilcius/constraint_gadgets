"""
add_to_vcg_database.py -- Train a VCG and add it to the gadget database.

Training procedure
------------------
1. Single QAOA run at p=1 (2 parameters, fast).
   Optimal angles are broadcast to ma-QAOA format to seed the first restart.

2. ma-QAOA layer sweep starting at p=1:
   - p=1 : first restart seeded from QAOA p=1 angles (broadcast γ → all
           num_gamma entries, β → all num_beta entries).  Subsequent restarts
           are random.
   - p>1 : joint re-optimisation of all p*(num_gamma+num_beta) parameters,
           warm-started from the previous depth's optimal angles.
   Stop when AR >= ar_threshold or max_layers is reached.

The best ma-QAOA gadget (at the threshold layer, or the best layer if
threshold is not met) is registered in the GadgetDatabase.

Usage (library)
---------------
    from run.add_to_vcg_database import train_and_add
    ar = train_and_add(
        constraints=['6*x_0 + 2*x_1 + 2*x_2 <= 3'],
        db_path='gadgets/gadget_db.pkl',
    )

Usage (CLI)
-----------
    python run/add_to_vcg_database.py \\
        --constraints "6*x_0 + 2*x_1 + 2*x_2 <= 3" \\
        --db gadgets/gadget_db.pkl

    # With explicit settings:
    python run/add_to_vcg_database.py \\
        --constraints "5*x_0 + 10*x_1 + 1*x_2 + 9*x_3 <= 19" \\
        --db gadgets/gadget_db.pkl \\
        --ar-threshold 0.999 --max-layers 8 \\
        --qaoa-restarts 5 --qaoa-steps 150 \\
        --ma-restarts 20 --ma-steps 200 \\
        --lr 0.05 --shots 10000

    # Write the full result row (for analytics) to a separate file:
    python run/add_to_vcg_database.py \\
        --constraints "6*x_0 + 2*x_1 + 2*x_2 <= 3" \\
        --db gadgets/gadget_db.pkl \\
        --result-out gadgets/pending/task_0.pkl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import re
import argparse

import pandas as pd
from pennylane import numpy as np

from core import vcg as vcg_module
from analyze_results.results_helper import collect_vcg_data
from core.vcg import find_in_db


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_and_add(
    constraints: list,
    db_path: str,
    ar_threshold: float = 0.999,
    max_layers: int = 8,
    qaoa_restarts: int = 5,
    qaoa_steps: int = 150,
    ma_restarts: int = 20,
    ma_steps: int = 200,
    lr: float = 0.05,
    shots: int = 10_000,
    result_out: str = None,
    verbose: bool = True,
) -> float:
    """Run a single QAOA p=1 to get warm-start angles, then sweep ma-QAOA
    layers until AR >= ar_threshold, and add the best gadget to the DB.

    Parameters
    ----------
    constraints : list[str]
        Zero-indexed constraint strings, e.g. ['6*x_0 + 2*x_1 + 2*x_2 <= 3'].
    db_path : str
        Path to the GadgetDatabase pickle.  Created if it does not exist.
    ar_threshold : float
        Stop training when AR >= this value.  Default 0.999.
    max_layers : int
        Maximum ma-QAOA layers before giving up.
    qaoa_restarts, qaoa_steps : int
        Optimisation budget for the single QAOA p=1 run.
    ma_restarts, ma_steps : int
        Optimisation budget per ma-QAOA layer.
    lr : float
        Adam learning rate.
    shots : int
        Measurement shots for the final counts circuit.
    result_out : str or None
        If given, save the full result row (all metrics) to this pickle.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    ar : float
        Best AR achieved by ma-QAOA.
    """
    # Check if already in DB
    if find_in_db(constraints, db_path, n_layers=1, angle_strategy='ma-QAOA'):
        if verbose:
            print(f'  [skip] Already in DB: {constraints[0][:60]}')
        return 1.0

    # Determine flag wires from the union of all constraint variables
    all_vars = set()
    for c in constraints:
        all_vars.update(int(m) for m in re.findall(r'x_(\d+)', c))
    n_vars = len(all_vars)
    flag_wires = list(range(n_vars, n_vars + len(constraints)))

    if verbose:
        print(f'\nConstraint(s): {[c[:60] for c in constraints]}')
        print(f'  n_vars={n_vars}  flag_wires={flag_wires}')

    # Probe once to get Hamiltonian structure (cheap: 1 step, 1 restart)
    probe = vcg_module.VCG(
        constraints=constraints, flag_wires=flag_wires,
        angle_strategy='ma-QAOA', decompose=True,
        n_layers=1, steps=1, num_restarts=1,
    )
    n_pauli = probe.num_gamma
    if verbose:
        n_good = probe.outcomes.count(-1)
        n_bad  = probe.outcomes.count(1)
        print(f'  States: {len(probe.outcomes)} total, {n_good} good, {n_bad} bad')
        print(f'  Pauli terms: {n_pauli}  (wires: {probe.num_beta})')

    # ── Step 1: single QAOA p=1 run ─────────────────────────────────────────
    if verbose:
        print(f'  QAOA p=1  (restarts={qaoa_restarts}, steps={qaoa_steps})')

    qaoa_gadget = vcg_module.VCG(
        constraints=constraints, flag_wires=flag_wires,
        angle_strategy='QAOA', decompose=False,
        n_layers=1, steps=qaoa_steps, num_restarts=qaoa_restarts, learning_rate=lr,
    )
    qaoa_cost, _ = qaoa_gadget.optimize_angles(qaoa_gadget.do_evolution_circuit)
    qaoa_ar = (float(qaoa_cost) - 1.0) / -2.0
    qaoa_angles = qaoa_gadget.opt_angles  # shape (1, 2)
    if verbose:
        print(f'    AR={qaoa_ar:.4f}  time={qaoa_gadget.optimize_time:.1f}s')

    # ── Step 2: ma-QAOA layer sweep ──────────────────────────────────────────
    if verbose:
        print(f'  ma-QAOA sweep  (restarts={ma_restarts}, steps={ma_steps})')

    best_ma_ar = 0.0
    best_ma_gadget = None
    prev_best_ma = None

    for p in range(1, max_layers + 1):
        gadget = vcg_module.VCG(
            constraints=constraints, flag_wires=flag_wires,
            angle_strategy='ma-QAOA', decompose=True,
            n_layers=p, steps=ma_steps, num_restarts=ma_restarts, learning_rate=lr,
        )

        if prev_best_ma is None:
            # p=1: seed first restart from QAOA p=1 optimal angles
            opt_cost, _ = gadget.optimize_angles(
                gadget.do_evolution_circuit,
                starting_angles_from_qaoa=qaoa_angles,
            )
        else:
            # p>1: joint re-opt all layers, warm-started from previous depth
            opt_cost, _ = gadget.optimize_angles(
                gadget.do_evolution_circuit,
                prev_layer_angles=prev_best_ma,
            )

        prev_best_ma = gadget.opt_angles
        ar = (float(opt_cost) - 1.0) / -2.0
        if ar > best_ma_ar:
            best_ma_ar = ar
            best_ma_gadget = gadget

        if verbose:
            mark = '✓' if ar >= ar_threshold else ' '
            print(f'    {mark} p={p}: AR={ar:.4f}  time={gadget.optimize_time:.1f}s')
        if ar >= ar_threshold:
            break

    if best_ma_ar < ar_threshold and verbose:
        print(f'  [warn] Did not reach AR>={ar_threshold} within {max_layers} layers. '
              f'Best ma-QAOA AR={best_ma_ar:.4f}')

    # ── Collect and store ────────────────────────────────────────────────────
    row = collect_vcg_data(
        best_ma_gadget,
        constraint_type='',
        skip_optimize=True,
        shots=shots,
        gadget_db_path=db_path,
    )

    if result_out:
        os.makedirs(os.path.dirname(result_out) if os.path.dirname(result_out) else '.', exist_ok=True)
        pd.DataFrame([row]).to_pickle(result_out)
        if verbose:
            print(f'  Result saved to {result_out}')

    if verbose:
        print(f'  Added to DB: {db_path}  (AR={best_ma_ar:.4f}, '
              f'p={best_ma_gadget.n_layers}, strategy=ma-QAOA)')

    return best_ma_ar


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train a VCG and add it to the gadget database.'
    )
    p.add_argument('--constraints', nargs='+', required=True,
                   help='Zero-indexed constraint strings (one or more).')
    p.add_argument('--db', default='gadgets/gadget_db.pkl',
                   help='Path to the GadgetDatabase pickle.')
    p.add_argument('--ar-threshold', type=float, default=0.999)
    p.add_argument('--max-layers', type=int, default=8)
    p.add_argument('--qaoa-restarts', type=int, default=5)
    p.add_argument('--qaoa-steps', type=int, default=150)
    p.add_argument('--ma-restarts', type=int, default=20)
    p.add_argument('--ma-steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--shots', type=int, default=10_000)
    p.add_argument('--result-out', default=None,
                   help='Path to save the full result row pickle (optional).')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    train_and_add(
        constraints=args.constraints,
        db_path=args.db,
        ar_threshold=args.ar_threshold,
        max_layers=args.max_layers,
        qaoa_restarts=args.qaoa_restarts,
        qaoa_steps=args.qaoa_steps,
        ma_restarts=args.ma_restarts,
        ma_steps=args.ma_steps,
        lr=args.lr,
        shots=args.shots,
        result_out=args.result_out,
    )
