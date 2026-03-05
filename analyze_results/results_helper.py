"""
results_helper.py -- Shared experiment utilities for constraint_gadget.

Provides:
    ResultsCollector                -- incremental result accumulation and pickle persistence
    GadgetDatabase                  -- lightweight VCG database (minimal fields for HybridQAOA)
    read_typed_csv(filepath)        -- parse n_vars; ['c1', ...] CSV format
    remap_to_zero_indexed(...)      -- canonicalise constraint to x_0, x_1, ...
    remap_constraint_to_vars(...)   -- embed zero-indexed constraint into QUBO positions
    collect_vcg_data(...)           -- collect metrics from a trained VCG instance
    collect_hybrid_data(...)        -- collect metrics from a HybridQAOA instance
    collect_penalty_data(...)       -- collect metrics from a PenaltyQAOA instance
"""

import os
import re

import pandas as pd
import pennylane as qml
from pennylane import numpy as np

from core import vcg as vcg_module
from core import hybrid_qaoa as hq
from core import penalty_qaoa as pq


# ─────────────────────────────────────────────────────────────────────────────
# ResultsCollector
# ─────────────────────────────────────────────────────────────────────────────

class ResultsCollector:
    """Accumulate per-experiment result dicts and persist to pickle.

    Usage
    -----
        collector = ResultsCollector()
        collector.load('results/my_run.pkl')   # resume if exists
        collector.add(row_dict)                # append one experiment row
        collector.save('results/my_run.pkl')   # persist to pickle
        df = collector.to_dataframe()
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, row_dict: dict) -> None:
        """Append a single result dict (one experiment row)."""
        self._rows.append(row_dict)

    def load(self, path: str) -> None:
        """Resume from an existing pickle, prepending any rows already held."""
        if os.path.exists(path):
            existing = pd.read_pickle(path)
            if not existing.empty:
                self._rows = existing.to_dict('records') + self._rows

    def save(self, path: str) -> None:
        """Save all accumulated rows as a pandas pickle."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.to_dataframe().to_pickle(path)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all accumulated rows as a DataFrame."""
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)


# ─────────────────────────────────────────────────────────────────────────────
# GadgetDatabase
# ─────────────────────────────────────────────────────────────────────────────

class GadgetDatabase:
    """Lightweight store of pre-trained VCG gadgets for use in HybridQAOA.

    Retains only the fields required by VCG.get_pre_made_data / vcg.find_in_db:
        constraints, n_layers, angle_strategy, outcomes, Hamiltonian, opt_angles

    All analysis fields (counts, resources, timing, AR, …) are intentionally
    excluded to keep the database small.  Entries are deduplicated on
    (constraints, n_layers, angle_strategy) so repeated runs don't grow the file.

    Usage
    -----
        db = GadgetDatabase()
        db.load('gadgets/gadget_db.pkl')   # extend with any existing entries
        db.add(vcg_row)                    # register a new gadget from a result row
        db.save('gadgets/gadget_db.pkl')   # persist
    """

    # Fields extracted from a full VCG result row.
    _FIELDS = ['constraints', 'n_layers', 'angle_strategy',
               'outcomes', 'Hamiltonian', 'opt_angles']

    def __init__(self) -> None:
        self._rows: list[dict] = []

    # ------------------------------------------------------------------

    def add(self, vcg_row: dict) -> None:
        """Extract minimal fields from a full VCG result row and register.

        Silently skips if an entry with the same (constraints, n_layers,
        angle_strategy) already exists in the database.

        Parameters
        ----------
        vcg_row : dict
            A single-row result dict as returned by collect_vcg_data.
            Values may be bare scalars or length-1 lists (both accepted).
        """
        def _unpack(v):
            return v[0] if isinstance(v, list) and len(v) == 1 else v

        entry = {k: _unpack(vcg_row[k]) for k in self._FIELDS}

        # Deduplication check
        for existing in self._rows:
            if (set(existing['constraints']) == set(entry['constraints'])
                    and existing['n_layers'] == entry['n_layers']
                    and existing['angle_strategy'] == entry['angle_strategy']):
                return   # already present

        self._rows.append(entry)

    def load(self, path: str) -> None:
        """Extend with entries from an existing gadget database pickle."""
        if os.path.exists(path):
            existing = pd.read_pickle(path)
            if not existing.empty:
                self._rows = existing.to_dict('records') + self._rows

    def save(self, path: str) -> None:
        """Persist all entries to a pickle file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.to_dataframe().to_pickle(path)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all entries as a DataFrame."""
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


# ─────────────────────────────────────────────────────────────────────────────
# Constraint string helpers
# ─────────────────────────────────────────────────────────────────────────────

def remap_to_zero_indexed(constraint_str: str, variables: set) -> tuple:
    """Remap variable indices in a constraint string to 0-based.

    Produces a canonical form with x_0, x_1, ... so that VCG training
    uses 0-indexed wires.  ConstraintMapper recovers the original variable
    assignment via permutation search when loading from a gadget database.

    Returns
    -------
    remapped : str
        Constraint string with variables renamed x_0, x_1, ...
    n_c_vars : int
        Number of distinct variables (= flag wire offset for training).
    """
    var_list = sorted(variables)
    var_map = {old: new for new, old in enumerate(var_list)}

    def _replace(m):
        idx = int(m.group(1))
        return f'x_{var_map[idx]}' if idx in var_map else m.group(0)

    return re.sub(r'x_(\d+)', _replace, constraint_str), len(var_list)


def remap_constraint_to_vars(constraint_str: str, target_indices: list) -> str:
    """Embed a zero-indexed constraint into specific QUBO variable positions.

    Maps x_0 → x_{target_indices[0]}, x_1 → x_{target_indices[1]}, etc.
    Used to place a constraint on an arbitrary subset of a larger QUBO.

    Parameters
    ----------
    constraint_str : str
        Zero-indexed constraint, e.g. 'x_0 + x_1 + x_2 == 1'.
    target_indices : list[int]
        QUBO variable indices to assign, e.g. [2, 3, 4].

    Returns
    -------
    str
        Remapped constraint, e.g. 'x_2 + x_3 + x_4 == 1'.
    """
    def _replace(m):
        idx = int(m.group(1))
        return f'x_{target_indices[idx]}'
    return re.sub(r'x_(\d+)', _replace, constraint_str)


def read_typed_csv(filepath: str) -> list:
    """Parse a typed constraint CSV into a list of (n_vars, constraints) tuples.

    Expected line format:
        n_vars; ['constraint_string_1', 'constraint_string_2', ...]

    Returns
    -------
    list of (int, list[str])
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_str, rest = line.split('; ', 1)
            n_vars = int(n_str)
            parsed = eval(rest)
            if isinstance(parsed, str):
                constraints = [parsed]
            elif isinstance(parsed, tuple):
                constraints = list(parsed)
            else:
                constraints = list(parsed)
            rows.append((n_vars, constraints))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Result collectors
# ─────────────────────────────────────────────────────────────────────────────

def collect_vcg_data(vcg: vcg_module.VCG, combined: bool = False,
                     single_flag: bool = False, decompose: bool = True,
                     constraint_type: str = '',
                     gadget_db_path: str = None) -> dict:
    """Train a VCG and collect metrics.

    Calls optimize_angles → get_circuit_resources → do_counts_circuit internally.

    If ``gadget_db_path`` is provided the minimal gadget entry (constraints,
    n_layers, angle_strategy, outcomes, Hamiltonian, opt_angles) is appended to
    the GadgetDatabase at that path immediately after optimisation.  Duplicate
    entries are silently skipped.

    Returns a single-row dict suitable for pd.DataFrame() or ResultsCollector.
    """
    C_max = float(max(qml.eigvals(vcg.constraint_Ham)))
    C_min = float(min(qml.eigvals(vcg.constraint_Ham)))
    opt_cost, opt_angles = vcg.optimize_angles(vcg.do_evolution_circuit)
    resources, est_shots, est_error, group_est_shots, group_est_error = vcg.get_circuit_resources()
    counts = vcg.do_counts_circuit(shots=est_shots)
    row = {
        'constraint_type': [constraint_type],
        'constraints': [vcg.constraints],
        'n_x': [vcg.n_x],
        'n_c': [len(vcg.constraints)],
        'combined': [combined],
        'single_flag': [single_flag],
        'decompose': [decompose],
        'Hamiltonian': [vcg.constraint_Ham],
        'outcomes': [vcg.outcomes],
        'angle_strategy': [vcg.angle_strategy],
        'n_layers': [vcg.n_layers],
        'num_gamma': [vcg.num_gamma],
        'num_beta': [vcg.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [float(opt_cost)],
        'counts': [counts],
        'resources': [resources],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [vcg.hamiltonian_time],
        'optimize_time': [vcg.optimize_time],
        'counts_time': [vcg.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'AR': [(float(opt_cost) - C_max) / (C_min - C_max)],
    }
    if gadget_db_path is not None:
        db = GadgetDatabase()
        db.load(gadget_db_path)
        db.add(row)
        db.save(gadget_db_path)
    return row


def collect_hybrid_data(constraints: list, hybrid: hq.HybridQAOA,
                        qubo_string: str, min_val: float = None,
                        previous_angles=None,
                        constraint_type: str = '',
                        var_assignment: list = None) -> dict:
    """Run HybridQAOA and collect metrics.

    Calls solve() for single-layer runs (VCG training is handled automatically
    inside HybridQAOA via gadget_db_path lookup), or optimize_angles with
    prev_layer_angles for multi-layer warm-starting.

    Returns a single-row dict suitable for pd.DataFrame() or ResultsCollector.
    """
    C_max = float(max(qml.eigvals(hybrid.problem_ham)))
    C_min = float(min(qml.eigvals(hybrid.problem_ham)))
    if previous_angles is not None:
        opt_cost, opt_angles = hybrid.optimize_angles(
            hybrid.do_evolution_circuit, prev_layer_angles=previous_angles
        )
        counts = hybrid.do_counts_circuit(shots=10_000)
    else:
        opt_cost, counts, opt_angles = hybrid.solve()
    _, est_shots, est_error, group_est_shots, group_est_error = hybrid.get_circuit_resources()
    return {
        'constraint_type': [constraint_type],
        'qubo_string': [qubo_string],
        'constraints': [constraints],
        'var_assignment': [var_assignment],
        'n_x': [hybrid.n_x],
        'n_c': [len(constraints)],
        'Hamiltonian': [hybrid.problem_ham],
        'angle_strategy': [hybrid.angle_strategy],
        'mixer': [hybrid.mixer],
        'n_layers': [hybrid.n_layers],
        'num_gamma': [hybrid.num_gamma],
        'num_beta': [hybrid.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [float(opt_cost)],
        'counts': [counts],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [getattr(hybrid, 'hamiltonian_time', None)],
        'optimize_time': [hybrid.optimize_time],
        'counts_time': [hybrid.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'min_val': [min_val],
        'AR': [(float(opt_cost) - C_max) / (C_min - C_max)],
    }


def collect_penalty_data(constraints: list, penalty: pq.PenaltyQAOA,
                         qubo_string: str, min_val: float = None,
                         constraint_type: str = '',
                         var_assignment: list = None) -> dict:
    """Run PenaltyQAOA and collect metrics.

    Calls optimize_angles → get_circuit_resources → do_counts_circuit internally.

    Returns a single-row dict suitable for pd.DataFrame() or ResultsCollector.
    """
    C_max = float(max(qml.eigvals(penalty.full_Ham)))
    C_min = float(min(qml.eigvals(penalty.full_Ham)))
    opt_cost, opt_angles = penalty.optimize_angles(penalty.do_evolution_circuit)
    _, est_shots, est_error, group_est_shots, group_est_error = penalty.get_circuit_resources()
    counts = penalty.do_counts_circuit(shots=10_000)
    return {
        'constraint_type': [constraint_type],
        'qubo_string': [qubo_string],
        'constraints': [constraints],
        'var_assignment': [var_assignment],
        'n_x': [penalty.n_x],
        'n_c': [len(constraints)],
        'Hamiltonian': [penalty.full_Ham],
        'angle_strategy': [penalty.angle_strategy],
        'penalty': [penalty.penalty_param],
        'n_layers': [penalty.n_layers],
        'num_gamma': [penalty.num_gamma],
        'num_beta': [penalty.num_beta],
        'opt_angles': [opt_angles.tolist()],
        'opt_cost': [float(opt_cost)],
        'counts': [counts],
        'est_shots': [est_shots],
        'est_error': [est_error],
        'group_est_shots': [group_est_shots],
        'group_est_error': [group_est_error],
        'hamiltonian_time': [getattr(penalty, 'hamiltonian_time', None)],
        'optimize_time': [penalty.optimize_time],
        'counts_time': [penalty.count_time],
        'C_max': [C_max],
        'C_min': [C_min],
        'min_val': [min_val],
        'AR': [(float(opt_cost) - C_max) / (C_min - C_max)],
    }
