"""
compute_vcg_resources.py -- Analytical gate-count table for trained VCGs.

Loads the VCG training database (gadgets/vcg_db.pkl) and analytically counts
gates for one VCG.opt_circuit() call per gadget.  No re-training is performed.

VCG circuit structure (general QAOA path):
    n_x Hadamard          (initialisation per call)
    n_layers × [
        num_gamma MultiRZ(k_i)   cost unitary (k_i = wire count of term i)
        n_x RX                   X-mixer
    ]

Special cases detected in VCG.__init__ (no QAOA):
    single feasible state  → X gates only
    Dicke superposition    → multiweight Dicke state prep

Gate keys used
--------------
Hadamard, RX, RZ, CNOT — MultiRZ(k) is decomposed as 2(k-1) CNOT + 1 RZ
for consistency with the Hybrid/Penalty gate basis.

Output DataFrame columns
------------------------
vcg_key          str    normalised constraint string (DB lookup key)
constraint_type  str    family name (knapsack / quadratic_knapsack)
constraints      list   original constraint string(s)
n_x              int    number of decision variables
n_layers         int    QAOA layers used (from saved training data)
sp_total         int    total gates for one opt_circuit() call
sp_1q            int    one-qubit gate count
sp_2q            int    two-qubit gate count
sp_gates         dict   {gate_name: count}

Usage
-----
    python analyze_results/compute_vcg_resources.py
    python analyze_results/compute_vcg_resources.py \\
        --db      gadgets/vcg_db.pkl \\
        --output  results/vcg_circuit_resources
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pennylane as qml
from tqdm import tqdm

from core.vcg import VCG
from core.resource_estimation import (
    count_dicke_gates,
    count_leq_gates,
    _1Q_GATES_SET,
    _2Q_GATES_SET,
)


# ── Constraint family classification ─────────────────────────────────────────

def _classify(key: str) -> str:
    """Classify a constraint key as 'knapsack' or 'quadratic_knapsack'."""
    return 'quadratic_knapsack' if re.search(r'x_\d+\*x_\d+', key) else 'knapsack'


# ── Gate classification ───────────────────────────────────────────────────────

def _count_1q_2q(gate_dict: dict):
    n1 = sum(cnt for name, cnt in gate_dict.items() if name in _1Q_GATES_SET)
    n2 = sum(cnt for name, cnt in gate_dict.items() if name in _2Q_GATES_SET)
    return n1, n2


# ── VCG gate counting ─────────────────────────────────────────────────────────

def _count_multiweight_dicke_gates(n_x: int, weights: list) -> dict:
    """Gate counts for prepare_dicke_multiweight_state(wires, weights)."""
    if not weights:
        return {}
    return count_leq_gates(n_x, max(weights))


def count_vcg_opt_circuit_gates(vcg: VCG, n_layers_override: int | None = None) -> dict:
    """
    Analytical gate count for one VCG.opt_circuit() call.

    Parameters
    ----------
    vcg : VCG
        Instantiated VCG (no training needed — special-case flags are set in __init__).
    n_layers_override : int or None
        If provided, overrides vcg.n_layers.  Pass the n_layers from the DB entry
        when calling with a freshly instantiated (untrained) VCG.
    """
    use_qaoa = n_layers_override is not None and n_layers_override >= 1

    if not use_qaoa:
        if vcg._single_feasible_bitstring is not None:
            x_count = vcg._single_feasible_bitstring.count('1')
            return {"X": x_count} if x_count > 0 else {}
        if vcg._dicke_superposition_weights is not None:
            return _count_multiweight_dicke_gates(vcg.n_x, vcg._dicke_superposition_weights)

    n_x = vcg.n_x
    n_layers = n_layers_override if n_layers_override is not None else vcg.n_layers

    if n_layers is None or n_layers == 0:
        return {"Hadamard": n_x}

    gates: dict[str, int] = {"Hadamard": n_x}

    cnot_per_layer = 0
    rz_per_layer = 0
    _, ops = vcg.constraint_Ham.terms()
    for op in ops:
        s = qml.pauli.pauli_word_to_string(op)
        if re.search(r'^I+$', s):
            continue
        k = len(op.wires)
        cnot_per_layer += 2 * max(0, k - 1)
        rz_per_layer += 1

    gates["CNOT"] = cnot_per_layer * n_layers
    gates["RZ"]   = rz_per_layer * n_layers
    gates["RX"]   = n_x * n_layers

    return {k: v for k, v in gates.items() if v > 0}


# ── Main builder ──────────────────────────────────────────────────────────────

def build_vcg_resource_table(
    db_path: str = "gadgets/vcg_db.pkl",
    output_prefix: str = "results/vcg_circuit_resources",
) -> pd.DataFrame:
    """
    Compute per-VCG circuit resources from the training database.

    Parameters
    ----------
    db_path : str
        Path to vcg_db.pkl (dict keyed by normalised constraint string).
    output_prefix : str
        Prefix for output files (.pkl and .csv).  Pass None to skip saving.

    Returns
    -------
    pd.DataFrame  (one row per trained VCG)
    """
    print(f"Loading VCG database from {db_path} ...")
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    print(f"  {len(db)} VCG entries")

    rows = []
    n_failed = 0

    for vcg_key, entry in tqdm(db.items(), desc="Counting VCG gates"):
        try:
            constraints = list(entry['constraints'])
            n_x         = int(entry['n_x'])
            n_layers    = int(entry['n_layers'])
            ctype       = entry.get('family') or _classify(vcg_key)

            vcg = VCG(constraints)
            gate_dict = count_vcg_opt_circuit_gates(vcg, n_layers_override=n_layers)
            n1, n2 = _count_1q_2q(gate_dict)

            rows.append({
                "vcg_key":         vcg_key,
                "constraint_type": ctype,
                "constraints":     constraints,
                "n_x":             n_x,
                "n_layers":        n_layers,
                "sp_total":        sum(gate_dict.values()),
                "sp_1q":           n1,
                "sp_2q":           n2,
                "sp_gates":        gate_dict,
            })

        except Exception as e:
            n_failed += 1
            tqdm.write(f"  [WARN] {vcg_key[:60]} failed: {e}")
            if os.environ.get("DEBUG"):
                traceback.print_exc()

    df = pd.DataFrame(rows)
    print(f"\nDone. {len(df)} rows, {n_failed} failures.")

    if not df.empty:
        print(f"  sp_total range: {df['sp_total'].min()}–{df['sp_total'].max()}")

    if output_prefix:
        os.makedirs(
            os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else ".",
            exist_ok=True,
        )
        df.to_pickle(output_prefix + ".pkl")
        df.to_csv(output_prefix + ".csv", index=False)
        print(f"  Saved → {output_prefix}.pkl / .csv")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db",     default="gadgets/vcg_db.pkl",
                        help="Path to vcg_db.pkl (default: gadgets/vcg_db.pkl)")
    parser.add_argument("--output", default="results/vcg_circuit_resources",
                        help="Output file prefix (no extension)")
    args = parser.parse_args()

    build_vcg_resource_table(db_path=args.db, output_prefix=args.output)


if __name__ == "__main__":
    main()
