"""
compute_circuit_resources.py -- Gate-count resource table for all experiments.

Iterates over every task in experiment_params_overlapping.jsonl and computes
circuit resources analytically (no JAX JIT, no circuit execution) using
pennylane.estimator (qre) circuit tracing, for both a NISQ and an FTQC gate
set (see core/resource_estimation.py for the gate-set definitions and why
each is what it is).

The key identity exploited is:

    total_gates(p) = r_sp + p * r_layer

where
  r_sp    = state preparation (once): structural/VCG state preps + slack |+> init
  r_layer = one QAOA layer:  cost unitary + mixer
            (for Grover, each layer also applies 2x state prep inside the mixer)

Both r_sp and r_layer are computed independently of p, so the table stores them
separately and total gate counts for any p can be derived without re-running.
VCG gadgets are traced directly from the saved training database (opt_angles /
n_layers / constraint_Ham) and are already included in sp/layer counts -- there
is no separate VCG gate-count file or column anymore.

Input files
-----------
  run/params/experiment_params_overlapping.jsonl
      One JSON line per experiment: constraints, n_x, qubo_idx, families.

  data/qubos.csv
      QUBO matrices, loaded via data.make_data.read_qubos_from_file.

  gadgets/vcg_db.pkl  (optional)
      Raw VCG training database: {normalized_constraint: {opt_angles,
      n_layers, single_feasible_bitstring, dicke_superposition_weights}}.
      When provided, VCG-classified structural constraints are traced from
      their trained circuit; otherwise their contribution is zero and
      vcg_missing_pc is flagged True.

Output DataFrame
----------------
Saved to results/circuit_resources.pkl (and .csv).  One row per experiment.
Every gate-count column below is reported twice: unsuffixed for the NISQ
gate set, and with an "_ftqc" suffix for the FTQC gate set.

Identity columns
~~~~~~~~~~~~~~~~
  task_id          int      1-indexed position in experiment_params_overlapping.jsonl
  constraints_hash str      sorted(constraints) stringified — matches build_problem_table
  constraint_type  str      '+'.join(families)
  n_x              int      number of decision variables
  n_c              int      number of constraints
  qubo_string      str      QUBO identifier string
  families         list     constraint family names

PC-QAOA columns  (per-gadget mixer: XY for Dicke/flow, Grover for VCG/LEQ, X for free qubits)
~~~~~~~~~~~~~~~~~~~~~
  n_qubits_pc         int      total qubits = n_x + n_slack_pc
  n_slack_pc          int      slack qubits from penalty constraints
  n_struct_pc         int      number of structural (state-prep) constraints
  n_pen_pc            int      number of penalty constraints going into Hamiltonian
  has_vcg_pc          bool     any VCG-classified structural constraints?
  vcg_missing_pc      bool     VCG present but absent from the training db?

  sp_total_pc[_ftqc]       int   total gates in state prep (r_sp)
  sp_gates_pc[_ftqc]       dict  gate breakdown of r_sp
  layer_total_pc[_ftqc]    int   total gates in one QAOA layer (r_layer)
  layer_gates_pc[_ftqc]    dict  gate breakdown of r_layer
  sum_of_parts_pc[_ftqc]       int   upper-bound: each structural constraint's own
                                     gadget cost (once) + each penalized constraint's
                                     own penalty-term cost (x n_layers); see
                                     estimate_from_task's docstring
  sum_of_parts_pc_gates[_ftqc] dict  gate breakdown of the above

Penalty QAOA columns  (X mixer, all constraints penalised into Hamiltonian)
~~~~~~~~~~~~~~~~~~~~~
  n_qubits_p       int      total qubits = n_x + n_slack_p
  n_slack_p        int      slack qubits from all constraints

  sp_total_p[_ftqc]      int   total gates in state prep (= Hadamard on slack qubits)
  sp_gates_p[_ftqc]      dict  gate breakdown of r_sp
  layer_total_p[_ftqc]   int   total gates in one QAOA layer
  layer_gates_p[_ftqc]   dict  gate breakdown of r_layer
  sum_of_parts_p[_ftqc]       int   upper bound: every constraint's own penalty-term
                                    cost (x n_layers), summed
  sum_of_parts_p_gates[_ftqc] dict  gate breakdown of the above

Usage
-----
    cd /home/vilcius/Papers/constraint_gadget/code
    python analyze_results/compute_circuit_resources.py \\
        --params run/params/experiment_params_overlapping.jsonl \\
        --data   data/ \\
        --output results/circuit_resources \\
        --vcg-db gadgets/vcg_db.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

from data.make_data import read_qubos_from_file, get_optimal_x
from core.resource_estimation import estimate_from_task, _1Q_GATES_SET, _2Q_GATES_SET
from core import constraint_handler as ch


def _count_1q_2q(gate_dict: dict):
    n1 = sum(cnt for name, cnt in gate_dict.items() if name in _1Q_GATES_SET)
    n2 = sum(cnt for name, cnt in gate_dict.items() if name in _2Q_GATES_SET)
    return n1, n2


def load_vcg_db(vcg_db_path: str) -> dict:
    """
    Load the raw VCG training database.

    Returns {normalized_constraint: {"opt_angles":, "n_layers":,
    "single_feasible_bitstring":, "dicke_superposition_weights":}}, exactly
    the format `pc_qaoa._load_vcg_gadget` and `estimate_from_task` expect --
    passed straight through, no precomputed gate-count file needed.
    """
    with open(vcg_db_path, "rb") as f:
        return pickle.load(f)


def _row_for_gate_set(res_gs: dict, suffix: str) -> dict:
    """Flatten one gate set's slice of estimate_from_task's result into row columns."""
    sp_gates      = dict(res_gs["pc_qaoa_sp"].gate_counts)
    layer_gates   = dict(res_gs["pc_qaoa_layer"].gate_counts)
    p_sp_gates    = dict(res_gs["penalty_sp"].gate_counts)
    p_layer_gates = dict(res_gs["penalty_layer"].gate_counts)

    sp_1q, sp_2q = _count_1q_2q(sp_gates)
    layer_1q, layer_2q = _count_1q_2q(layer_gates)
    p_sp_1q, p_sp_2q = _count_1q_2q(p_sp_gates)
    p_layer_1q, p_layer_2q = _count_1q_2q(p_layer_gates)

    sop_pc = res_gs["sum_of_parts_pc_qaoa"]
    sop_p = res_gs["sum_of_parts_penalty"]

    return {
        f"sp_total_pc{suffix}":          res_gs["pc_qaoa_sp"].total_gates,
        f"sp_1q_pc{suffix}":             sp_1q,
        f"sp_2q_pc{suffix}":             sp_2q,
        f"sp_gates_pc{suffix}":          sp_gates,
        f"layer_total_pc{suffix}":       res_gs["pc_qaoa_layer"].total_gates,
        f"layer_1q_pc{suffix}":          layer_1q,
        f"layer_2q_pc{suffix}":          layer_2q,
        f"layer_gates_pc{suffix}":       layer_gates,
        f"sum_of_parts_pc{suffix}":      sum(sop_pc.values()),
        f"sum_of_parts_pc_gates{suffix}": sop_pc,

        f"sp_total_p{suffix}":          res_gs["penalty_sp"].total_gates,
        f"sp_1q_p{suffix}":             p_sp_1q,
        f"sp_2q_p{suffix}":             p_sp_2q,
        f"sp_gates_p{suffix}":          p_sp_gates,
        f"layer_total_p{suffix}":       res_gs["penalty_layer"].total_gates,
        f"layer_1q_p{suffix}":          p_layer_1q,
        f"layer_2q_p{suffix}":          p_layer_2q,
        f"layer_gates_p{suffix}":       p_layer_gates,
        f"sum_of_parts_p{suffix}":      sum(sop_p.values()),
        f"sum_of_parts_p_gates{suffix}": sop_p,
    }


def build_resource_table(
    params_path: str,
    data_dir: str = "data/",
    output_prefix: str = "results/circuit_resources",
    vcg_db_path: str | None = None,
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
    vcg_db_path : str or None
        Path to the raw VCG training database (gadgets/vcg_db.pkl).  When
        provided, VCG-classified structural constraints are traced from
        their trained circuit and included in the PC-QAOA sp/layer totals.

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
    if vcg_db_path and os.path.exists(vcg_db_path):
        print(f"Loading VCG database from {vcg_db_path} ...")
        vcg_db = load_vcg_db(vcg_db_path)
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
                penalty_weight=penalty_weight,
                vcg_db=vcg_db,
            )

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
                # ── PC-QAOA / Penalty shared metadata ────────────────────────
                "n_qubits_pc":       res["n_qubits_pc"],
                "n_slack_pc":        res["n_slack_pc"],
                "n_struct_pc":       len(si),
                "n_pen_pc":          len(pi),
                "has_vcg_pc":        res["has_vcg_pc"],
                "vcg_missing_pc":    res["vcg_missing_pc"],
                "n_qubits_p":       res["n_qubits_p"],
                "n_slack_p":        res["n_slack_p"],
            }
            row.update(_row_for_gate_set(res["nisq"], suffix=""))
            row.update(_row_for_gate_set(res["ftqc"], suffix="_ftqc"))

            rows.append(row)

        except Exception as e:
            n_failed += 1
            tqdm.write(f"  [WARN] task {task_id} failed: {e}")
            if os.environ.get("DEBUG"):
                traceback.print_exc()

    df = pd.DataFrame(rows)

    print(f"\nDone. {len(df)} rows, {n_failed} failures.")
    if not df.empty:
        print(f"  PC-QAOA   qubits range: {df['n_qubits_pc'].min()}–{df['n_qubits_pc'].max()}")
        print(f"  Penalty  qubits range: {df['n_qubits_p'].min()}–{df['n_qubits_p'].max()}")
        print(f"  PC-QAOA   sp gates (nisq):     {df['sp_total_pc'].describe()[['min','mean','max']].to_dict()}")
        print(f"  PC-QAOA   layer gates (nisq):  {df['layer_total_pc'].describe()[['min','mean','max']].to_dict()}")
        print(f"  PC-QAOA   sp gates (ftqc):     {df['sp_total_pc_ftqc'].describe()[['min','mean','max']].to_dict()}")
        print(f"  PC-QAOA   layer gates (ftqc):  {df['layer_total_pc_ftqc'].describe()[['min','mean','max']].to_dict()}")
        print(f"  Penalty  sp gates (nisq):     {df['sp_total_p'].describe()[['min','mean','max']].to_dict()}")
        print(f"  Penalty  layer gates (nisq):  {df['layer_total_p'].describe()[['min','mean','max']].to_dict()}")

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
    parser.add_argument("--params",   default="run/params/experiment_params_overlapping.jsonl")
    parser.add_argument("--data",     default="data/")
    parser.add_argument("--output",   default="results/circuit_resources")
    parser.add_argument("--vcg-db",   default=None,
                        help="Path to the raw VCG training database gadgets/vcg_db.pkl "
                             "(enables tracing VCG-gadget resources)")
    args = parser.parse_args()

    build_resource_table(
        params_path=args.params,
        data_dir=args.data,
        output_prefix=args.output,
        vcg_db_path=args.vcg_db,
    )


if __name__ == "__main__":
    main()
