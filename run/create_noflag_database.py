"""
create_noflag_database.py -- Train VCGNoFlag gadgets for all knapsack /
quadratic-knapsack constraints in the project and save them to a database.

The database is a dict saved as a pickle:
    { constraint_str: {
        'constraints':   list[str],   # zero-indexed constraint strings
        'opt_angles':    np.ndarray,  # trained ma-QAOA angles (or None)
        'n_layers':      int,
        'ar':            float,
        'entropy':       float,       # H_norm from exact statevector probs
        'n_x':           int,
        'single_feasible_bitstring':    str or None,
        'dicke_superposition_weights':  list[int] or None,
    } }

where constraint_str is the first (and usually only) element of 'constraints',
used as the canonical lookup key.

Entropy is computed from exact statevector probabilities (qml.probs) -- no
shot noise -- so entropy_threshold=0.9999 is a genuine stopping criterion.
If neither ar_threshold nor entropy_threshold is met within max_layers, the
best available state (highest entropy among AR-passing layers, or best AR
layer otherwise) is stored.

Usage
-----
    # Train all gadgets sequentially (default):
    python run/create_noflag_database.py

    # Custom thresholds / budget:
    python run/create_noflag_database.py \\
        --ar-threshold 0.999 --entropy-threshold 0.9999 --max-layers 8 \\
        --ma-restarts 20 --ma-steps 200 \\
        --db gadgets/noflag_db.pkl

    # Force-retrain even if already in DB:
    python run/create_noflag_database.py --force
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import pickle

import numpy as np

from core.vcg_no_flag import VCGNoFlag


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_db(db_path: str) -> dict:
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            return pickle.load(f)
    return {}


def _save_db(db: dict, db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)


def _db_key(constraints: list) -> str:
    """Canonical key: first constraint string (already zero-indexed)."""
    return constraints[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one(constraints: list, ar_threshold: float, entropy_threshold: float,
              max_layers: int, qaoa_restarts: int, qaoa_steps: int,
              ma_restarts: int, ma_steps: int, lr: float,
              samples: int, verbose: bool = True) -> dict:
    """Train a VCGNoFlag gadget and return a serialisable result dict."""
    vcg = VCGNoFlag(
        constraints=constraints,
        ar_threshold=ar_threshold,
        entropy_threshold=entropy_threshold,
        max_layers=max_layers,
        qaoa_restarts=qaoa_restarts,
        qaoa_steps=qaoa_steps,
        ma_restarts=ma_restarts,
        ma_steps=ma_steps,
        lr=lr,
        samples=samples,
    )
    vcg.train(verbose=verbose)
    return {
        'constraints':                  constraints,
        'opt_angles':                   vcg.opt_angles,
        'n_layers':                     vcg.n_layers,
        'ar':                           vcg.ar,
        'entropy':                      vcg.entropy,
        'n_x':                          vcg.n_x,
        'single_feasible_bitstring':    vcg._single_feasible_bitstring,
        'dicke_superposition_weights':  vcg._dicke_superposition_weights,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train VCGNoFlag gadgets for all knapsack/quadratic-knapsack constraints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--params', default='run/params/vcg_params.jsonl',
                        help='JSON-lines task file (from create_vcg_database.py --generate-params)')
    parser.add_argument('--db', default='gadgets/noflag_db.pkl',
                        help='Output database pickle path')
    parser.add_argument('--ar-threshold',      type=float, default=0.999)
    parser.add_argument('--entropy-threshold', type=float, default=0.9999)
    parser.add_argument('--max-layers',        type=int,   default=8)
    parser.add_argument('--qaoa-restarts',     type=int,   default=5)
    parser.add_argument('--qaoa-steps',        type=int,   default=150)
    parser.add_argument('--ma-restarts',       type=int,   default=20)
    parser.add_argument('--ma-steps',          type=int,   default=200)
    parser.add_argument('--lr',                type=float, default=0.05)
    parser.add_argument('--samples',           type=int,   default=10_000,
                        help='Shots for final counts circuit (entropy uses exact probs, not shots)')
    parser.add_argument('--force', action='store_true',
                        help='Retrain even if the gadget is already in the DB')
    args = parser.parse_args()

    if not os.path.exists(args.params):
        print(f'Params file not found: {args.params}')
        print('Generate it first with:')
        print('  python run/create_vcg_database.py --generate-params')
        sys.exit(1)

    with open(args.params) as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    print(f'Loaded {len(tasks)} tasks from {args.params}')

    db = _load_db(args.db)
    print(f'DB has {len(db)} existing entries.')

    skipped = 0
    trained = 0
    failed  = 0

    for i, task in enumerate(tasks):
        constraints = task['constraints']
        family      = task.get('family', '')
        key         = _db_key(constraints)

        print(f'\n[{i+1}/{len(tasks)}] {family}  {key[:70]}')

        if not args.force and key in db:
            entry = db[key]
            print(f'  [skip] Already in DB: AR={entry["ar"]:.4f}  '
                  f'entropy={entry["entropy"] if entry["entropy"] is not None else "N/A"}  '
                  f'layers={entry["n_layers"]}')
            skipped += 1
            continue

        try:
            result = train_one(
                constraints=constraints,
                ar_threshold=args.ar_threshold,
                entropy_threshold=args.entropy_threshold,
                max_layers=args.max_layers,
                qaoa_restarts=args.qaoa_restarts,
                qaoa_steps=args.qaoa_steps,
                ma_restarts=args.ma_restarts,
                ma_steps=args.ma_steps,
                lr=args.lr,
                samples=args.samples,
                verbose=True,
            )
            db[key] = result
            _save_db(db, args.db)   # save after every gadget in case of interruption
            ent_str = f'{result["entropy"]:.4f}' if result['entropy'] is not None else 'N/A'
            print(f'  Saved: AR={result["ar"]:.4f}  entropy={ent_str}  '
                  f'layers={result["n_layers"]}')
            trained += 1
        except Exception as e:
            import traceback
            print(f'  ERROR: {e}')
            traceback.print_exc()
            failed += 1

    print(f'\n{"="*60}')
    print(f'Done.  trained={trained}  skipped={skipped}  failed={failed}')
    print(f'DB now has {len(db)} entries. Saved to {args.db}')


if __name__ == '__main__':
    main()
