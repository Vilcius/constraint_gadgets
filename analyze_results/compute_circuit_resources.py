"""
compute_circuit_resources.py -- Gate-count resource table for all experiments.

Iterates over every task in experiment_params_overlapping.jsonl and computes circuit
resources analytically (no JAX JIT, no circuit execution) using
pennylane.estimator ResourceOperator subclasses.

The key identity exploited is:

    total_gates(p) = r_sp + p * r_layer

where
  r_sp    = state preparation (once): structural state preps + slack |+> init
  r_layer = one QAOA layer:  cost unitary + mixer
            (for Grover, each layer also applies 2× state prep inside the mixer)

Both r_sp and r_layer are computed independently of p, so the table stores them
separately and total gate counts for any p can be derived without re-running.

Input files
-----------
  run/params/experiment_params_overlapping.jsonl
      One JSON line per experiment: constraints, n_x, qubo_idx, families.

  data/qubos.csv
      QUBO matrices, loaded via data.make_data.read_qubos_from_file.

Output DataFrame
----------------
Saved to results/circuit_resources.pkl (and .csv).  One row per experiment.

Identity columns
~~~~~~~~~~~~~~~~
  task_id          int      1-indexed position in experiment_params_overlapping.jsonl
  constraints_hash str      sorted(constraints) stringified — matches build_problem_table
  constraint_type  str      '+'.join(families)
  n_x              int      number of decision variables
  n_c              int      number of constraints
  qubo_string      str      QUBO identifier string
  families         list     constraint family names

Hybrid QAOA columns  (mixer = Grover, structural constraints handled by state prep)
~~~~~~~~~~~~~~~~~~~~~
  n_qubits_h       int      total qubits = n_x + n_slack_h
  n_slack_h        int      slack qubits from penalty constraints
  n_struct_h       int      number of structural (state-prep) constraints
  n_pen_h          int      number of penalty constraints going into Hamiltonian

  sp_total_h       int      total gates in state prep (r_sp for Hybrid)
  sp_gates_h       dict     gate breakdown of r_sp  e.g. {"CNOT": 10, "CRY": 6, ...}

  layer_total_h    int      total gates in one QAOA layer (r_layer for Hybrid)
  layer_gates_h    dict     gate breakdown of r_layer

Penalty QAOA columns  (X mixer, all constraints penalised into Hamiltonian)
~~~~~~~~~~~~~~~~~~~~~
  n_qubits_p       int      total qubits = n_x + n_slack_p
  n_slack_p        int      slack qubits from all constraints

  sp_total_p       int      total gates in state prep (= Hadamard on slack qubits)
  sp_gates_p       dict     gate breakdown of r_sp

  layer_total_p    int      total gates in one QAOA layer
  layer_gates_p    dict     gate breakdown of r_layer

Usage
-----
    cd /home/vilcius/Papers/constraint_gadget/code
    python analyze_results/compute_circuit_resources.py \\
        --params run/params/experiment_params_overlapping.jsonl \\
        --data   data/ \\
        --output results/circuit_resources
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

from data.make_data import read_qubos_from_file, get_optimal_x
from core.resource_estimation import estimate_from_task, _1Q_GATES_SET, _2Q_GATES_SET
from core import constraint_handler as ch


def _resources_to_dict(res) -> dict:
    return {gate.name: int(count) for gate, count in res.gate_types.items()}


def _count_1q_2q(gate_dict: dict):
    n1 = sum(cnt for name, cnt in gate_dict.items() if name in _1Q_GATES_SET)
    n2 = sum(cnt for name, cnt in gate_dict.items() if name in _2Q_GATES_SET)
    return n1, n2


def _merge_gate_dicts(*dicts) -> dict:
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def load_vcg_db(vcg_resources_path: str) -> dict:
    """
    Load pre-computed VCG gate counts into a lookup dict.

    Returns {vcg_key: sp_gates_dict} where vcg_key = str(sorted(constraints)).
    When duplicate keys exist (same constraint, different training runs), the
    entry with the lowest sp_total is used (most efficient).
    """
    df = pd.read_pickle(vcg_resources_path)
    db = {}
    for _, row in df.iterrows():
        key = row["vcg_key"]
        gates = row["sp_gates"]
        total = row["sp_total"]
        if key not in db or total < sum(db[key].values()):
            db[key] = gates
    return db


def build_resource_table(
    params_path: str,
    data_dir: str = "data/",
    output_prefix: str = "results/circuit_resources",
    mixer: str = "Grover",
    vcg_resources_path: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-experiment circuit resources and return a DataFrame.

    Parameters
    ----------
    params_path : str
        Path to experiment_params_overlapping.jsonl.
    data_dir : str
        Directory containing qubos.csv.
    output_prefix : str
        Prefix for output files (saved as .pkl and .csv).
    mixer : str
        Hybrid mixer type (default "Grover").
    vcg_resources_path : str or None
        Path to vcg_circuit_resources.pkl.  When provided, VCG gate counts
        are included in hybrid sp/layer totals.

    Returns
    -------
    pd.DataFrame  (one row per experiment)
    """
    print(f"Loading tasks from {params_path} ...")
    with open(params_path) as f:
        tasks = [json.loads(line) for line in f]
    print(f"  {len(tasks)} tasks")

    print(f"Loading QUBOs from {data_dir} ...")
    qubos = read_qubos_from_file("qubos.csv", results_dir=data_dir)

    vcg_db = None
    if vcg_resources_path and os.path.exists(vcg_resources_path):
        print(f"Loading VCG database from {vcg_resources_path} ...")
        vcg_db = load_vcg_db(vcg_resources_path)
        print(f"  {len(vcg_db)} unique VCG keys loaded")
    else:
        print("  No VCG database provided — VCG gate counts will be zero.")

    rows = []
    n_failed = 0

    for task_id, task in enumerate(tqdm(tasks, desc="Estimating resources"), start=1):
        try:
            all_constraints = task["constraints"]
            n_x = task["n_x"]
            qubo_idx = task["qubo_idx"]
            families = task.get("families", [])
            qubo_string = qubos[n_x][qubo_idx]["qubo_string"]

            Q = qubos[n_x][qubo_idx]["Q"]
            _, _, total_min = get_optimal_x(Q, all_constraints)
            penalty_weight = float(5 + 2 * abs(total_min))

            res = estimate_from_task(
                task,
                qubos,
                n_layers=1,
                mixer=mixer,
                penalty_weight=penalty_weight,
                vcg_db=vcg_db,
            )

            # Non-VCG gate dicts from plre.estimate
            h_sp_plre   = _resources_to_dict(res["hybrid_sp"])
            h_lay_plre  = _resources_to_dict(res["hybrid_layer"])
            p_sp_gates  = _resources_to_dict(res["penalty_sp"])
            p_lay_gates = _resources_to_dict(res["penalty_layer"])

            # Merge VCG contributions into hybrid totals
            vcg_sp  = res["vcg_sp_gates"]
            vcg_lay = res["vcg_layer_gates"]
            h_sp_gates  = _merge_gate_dicts(h_sp_plre,  vcg_sp)
            h_lay_gates = _merge_gate_dicts(h_lay_plre, vcg_lay)

            h_sp_1q,  h_sp_2q  = _count_1q_2q(h_sp_gates)
            h_lay_1q, h_lay_2q = _count_1q_2q(h_lay_gates)
            p_sp_1q,  p_sp_2q  = _count_1q_2q(p_sp_gates)
            p_lay_1q, p_lay_2q = _count_1q_2q(p_lay_gates)

            # Constraint partition counts
            parsed = ch.parse_constraints(all_constraints)
            si, pi = ch.partition_constraints(parsed, strategy="auto")

            row = {
                # ── Identity ──────────────────────────────────────────────
                "task_id":          task_id,
                "constraints_hash": str(sorted(all_constraints)),
                "constraint_type":  "+".join(families),
                "n_x":              n_x,
                "n_c":              len(all_constraints),
                "qubo_string":      qubo_string,
                "families":         families,
                # ── Hybrid ────────────────────────────────────────────────
                "n_qubits_h":       res["n_qubits_h"],
                "n_slack_h":        res["n_slack_h"],
                "n_struct_h":       len(si),
                "n_pen_h":          len(pi),
                "has_vcg_h":        res["has_vcg_h"],
                "vcg_missing_h":    res["vcg_missing_h"],
                # state-prep (r_sp): non-VCG + VCG combined
                "sp_total_h":       sum(h_sp_gates.values()),
                "sp_1q_h":          h_sp_1q,
                "sp_2q_h":          h_sp_2q,
                "sp_gates_h":       h_sp_gates,
                "vcg_sp_gates_h":   vcg_sp,
                # one layer (r_layer): non-VCG + VCG combined
                "layer_total_h":    sum(h_lay_gates.values()),
                "layer_1q_h":       h_lay_1q,
                "layer_2q_h":       h_lay_2q,
                "layer_gates_h":    h_lay_gates,
                "vcg_layer_gates_h": vcg_lay,
                # ── Penalty ───────────────────────────────────────────────
                "n_qubits_p":       res["n_qubits_p"],
                "n_slack_p":        res["n_slack_p"],
                "sp_total_p":       res["penalty_sp"].total_gates,
                "sp_1q_p":          p_sp_1q,
                "sp_2q_p":          p_sp_2q,
                "sp_gates_p":       p_sp_gates,
                "layer_total_p":    res["penalty_layer"].total_gates,
                "layer_1q_p":       p_lay_1q,
                "layer_2q_p":       p_lay_2q,
                "layer_gates_p":    p_lay_gates,
            }

            rows.append(row)

        except Exception as e:
            n_failed += 1
            tqdm.write(f"  [WARN] task {task_id} failed: {e}")
            if os.environ.get("DEBUG"):
                traceback.print_exc()

    df = pd.DataFrame(rows)

    print(f"\nDone. {len(df)} rows, {n_failed} failures.")
    if not df.empty:
        print(f"  Hybrid   qubits range: {df['n_qubits_h'].min()}–{df['n_qubits_h'].max()}")
        print(f"  Penalty  qubits range: {df['n_qubits_p'].min()}–{df['n_qubits_p'].max()}")
        print(f"  Hybrid   sp gates:     {df['sp_total_h'].describe()[['min','mean','max']].to_dict()}")
        print(f"  Hybrid   layer gates:  {df['layer_total_h'].describe()[['min','mean','max']].to_dict()}")
        print(f"  Penalty  sp gates:     {df['sp_total_p'].describe()[['min','mean','max']].to_dict()}")
        print(f"  Penalty  layer gates:  {df['layer_total_p'].describe()[['min','mean','max']].to_dict()}")

    if output_prefix:
        os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else ".",
                    exist_ok=True)
        pkl_path = output_prefix + ".pkl"
        csv_path = output_prefix + ".csv"
        df.to_pickle(pkl_path)
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved pkl → {pkl_path}")
        print(f"  Saved csv → {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params",         default="run/params/experiment_params_overlapping.jsonl")
    parser.add_argument("--data",           default="data/")
    parser.add_argument("--output",         default="results/circuit_resources")
    parser.add_argument("--mixer",          default="Grover",
                        choices=["Grover", "X-Mixer", "XY", "Ring-XY"])
    parser.add_argument("--vcg-resources",  default=None,
                        help="Path to vcg_circuit_resources.pkl (enables VCG gate counting)")
    args = parser.parse_args()

    build_resource_table(
        params_path=args.params,
        data_dir=args.data,
        output_prefix=args.output,
        mixer=args.mixer,
        vcg_resources_path=args.vcg_resources,
    )


if __name__ == "__main__":
    main()
