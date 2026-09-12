"""resource_estimation.py

Circuit resource estimation for PC-QAOA and PenaltyQAOA, built on
``pennylane.estimator`` (``qre``).

Design
------
Rather than hand-deriving closed-form gate-count formulas, every circuit here
is a small plain-Python function that emits ``qre`` gate objects (``qre.CNOT``,
``qre.RY``, ``qre.Controlled(qre.RZ(), ...)``, ...) in *exactly* the same
control flow (loops / partition-tree traversal) as the real, executable
circuit it mirrors (see ``dicke_state_prep.py``, ``qaoa_base.py``, ``vcg.py``).
``qre.estimate(fn, gate_set=...)()`` then traces the function and tallies
gate counts automatically -- there is no separate combinatorial formula to
keep in sync with the real circuits.

Two target gate sets are supported and compared throughout:
    NISQ_GATE_SET -- {RX, RY, RZ, Hadamard, CNOT, X, Y, Z, T, S}
    FTQC_GATE_SET -- {Hadamard, T, CNOT, S}
``T`` is included in both sets purely so multi-controlled rotations
(doubly-controlled RY in the Dicke gadget, the multi-controlled-RZ Grover
reflection cascade, CSWAP in the Fredkin staircase) have a decomposition
path down to 1- and 2-qubit gates -- ``pennylane.estimator`` has no route for
those through rotations/CNOT alone.  ``S`` is included alongside it because,
without S available, the library's own Toffoli-based decomposition folds
every S it would otherwise use into 2 T gates (S = T^2) -- a purely notional
inflation of the (expensive, magic-state) T-count with no effect on any other
gate.  Confirmed empirically: adding S changes nothing for ordinary
rotations/single-controlled gates, only reduces T-count where multi-control
synthesis is already happening.  Everything else still resolves to plain
RX/RY/RZ/CNOT; T/S only show up where multi-control synthesis is genuinely
required.

For each constraint we estimate resources two ways (see
``estimate_constraint_resources`` / ``build_constraint_resource_db``):
  - "gadget"  -- used as an exact/VCG state-prep gadget for PC-QAOA
  - "penalty" -- penalized as a single term added to the cost Hamiltonian
so that when a full problem is solved, the per-constraint numbers can be
summed as a (conservative) upper bound and compared against the real,
transpiled full-circuit estimate (which can be smaller, since combining
Hamiltonian terms and transpilation both reduce gate counts below a naive
sum of independently-estimated parts).
"""
from __future__ import annotations

import re as _re
from typing import Dict, List, Tuple

import pennylane as qml
import pennylane.estimator as qre

from . import constraint_handler as ch
from .dicke_state_prep import _build_partition_tree


# ── Gate sets ──────────────────────────────────────────────────────────────

NISQ_GATE_SET = {"RX", "RY", "RZ", "Hadamard", "CNOT", "X", "Y", "Z", "T", "S"}
FTQC_GATE_SET = {"Hadamard", "T", "CNOT", "S"}
GATE_SETS = {"nisq": NISQ_GATE_SET, "ftqc": FTQC_GATE_SET}

# Exported gate-class sets (used by the analysis scripts to split 1q/2q counts).
_1Q_GATES_SET = frozenset({"RX", "RY", "RZ", "Hadamard", "X", "Y", "Z", "T", "S"})
_2Q_GATES_SET = frozenset({"CNOT"})


# ── Small helpers ────────────────────────────────────────────────────────────

def _closure(f, *args):
    """Zero-arg callable that calls ``f(*args)`` -- for handing to qre.estimate."""
    def _fn():
        return f(*args)
    return _fn


def _run_estimate(fn, gate_set) -> "qre.Resources":
    return qre.estimate(fn, gate_set=gate_set)()


def resources_to_dict(res: "qre.Resources") -> dict:
    """Plain-dict view of a qre.Resources object, suitable for pickling/saving."""
    return {
        "gate_counts": dict(res.gate_counts),
        "total_gates": res.total_gates,
        "algo_wires": res.algo_wires,
        "zeroed_wires": res.zeroed_wires,
        "any_state_wires": res.any_state_wires,
        "total_wires": res.total_wires,
    }


def _merge_gate_dicts(*dicts) -> dict:
    """Sum gate counts across multiple {gate_name: count} dicts."""
    merged: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def _pauli_term_sizes(hamiltonian: qml.Hamiltonian) -> Tuple[int, ...]:
    """Wire counts for each non-identity Pauli term in the Hamiltonian."""
    sizes = []
    _, ops = hamiltonian.terms()
    for op in ops:
        s = qml.pauli.pauli_word_to_string(op)
        if not _re.search(r"^I+$", s):
            sizes.append(len(op.wires))
    return tuple(sizes)


# ── qre-gate mirrors of the Dicke/LEQ/flow state-prep circuits ─────────────
#
# These replay the exact control flow of dicke_state_prep.py's circuit
# functions (same partition tree, same loop bounds -- ``_build_partition_tree``
# is imported and reused directly) but emit ``qre`` gate objects instead of
# ``qml`` gates.  Angle values are dropped throughout: none of the branches
# below are angle-*value*-dependent (the real code never skips a gate based
# on a computed rotation angle for the gadget families used in this codebase
# -- see the docstring of ``qre_prepare_dicke_multiweight_state``), only the
# combinatorial structure (tree shape, register sizes) matters for resource
# counting.

def _qre_gate_i(wires):
    qre.CNOT(wires=[wires[0], wires[1]])
    qre.CRY(wires=[wires[1], wires[0]])
    qre.CNOT(wires=[wires[0], wires[1]])


def _qre_gate_ii_l(wires):
    qre.CNOT(wires=[wires[0], wires[2]])
    qre.Controlled(qre.RY(), num_ctrl_wires=2, num_zero_ctrl=0,
                   wires=[wires[2], wires[1], wires[0]])
    qre.CNOT(wires=[wires[0], wires[2]])


def _qre_gate_scs_nk(k, wires):
    _qre_gate_i([wires[k - 1], wires[k]])
    for ell in range(2, k + 1):
        _qre_gate_ii_l([wires[k - ell], wires[k - ell + 1], wires[k]])


def _qre_scs_first_block(n, k, ell, wires):
    idxs = wires
    n_first, n_last = ell - k - 1, n - ell
    if n_first:
        idxs = idxs[n_first:]
    if n_last:
        idxs = idxs[:-n_last]
    _qre_gate_scs_nk(k, idxs)


def _qre_scs_second_block(n, k, ell, wires):
    idxs = wires
    n_last = n - ell
    if n_last:
        idxs = idxs[:-n_last]
    _qre_gate_scs_nk(ell - 1, idxs)


def _qre_dicke_state_scs(n, k, wires):
    if k == 0:
        return
    for ell in reversed(range(k + 1, n + 1)):
        _qre_scs_first_block(n, k, ell, wires)
    for ell in reversed(range(2, k + 1)):
        _qre_scs_second_block(n, k, ell, wires)


def _qre_reverse(register):
    for i in range(len(register) // 2):
        qre.SWAP(wires=[register[i], register[len(register) - 1 - i]])


def _qre_one_hot_encode(register):
    for i in range(len(register) - 1):
        qre.CNOT(wires=[register[i + 1], register[i]])


def _qre_revert_one_hot(register):
    for i in reversed(range(1, len(register))):
        qre.CNOT(wires=[register[i], register[i - 1]])


def _qre_controlled_addition(reg_a, reg_b):
    for ell in range(len(reg_a) - 1, -1, -1):
        lim = min(len(reg_b), ell + 1)
        for j in range(lim):
            if j == 0:
                qre.CRY(wires=[reg_a[ell], reg_b[j]])
            else:
                qre.Controlled(qre.RY(), num_ctrl_wires=2, num_zero_ctrl=0,
                               wires=[reg_a[ell], reg_b[j - 1], reg_b[j]])


def _qre_fredkin_stair(reg_a, reg_b):
    const = 1
    if len(reg_a) == len(reg_b):
        qre.CNOT(wires=[reg_b[-1], reg_a[-1]])
        const = 2
    for i in range(len(reg_b) - const, -1, -1):
        for j in range(i, len(reg_a) - 1):
            qre.CSWAP(wires=[reg_b[i], reg_a[j], reg_a[j + 1]])
        qre.CNOT(wires=[reg_b[i], reg_a[-1]])


def _qre_apply_wdb(v, k):
    reg_a = v.left_child.get_qubits()[:k]
    reg_b = v.right_child.get_qubits()[:k]
    _qre_one_hot_encode(reg_a)
    _qre_controlled_addition(reg_a, reg_b)
    _qre_revert_one_hot(reg_a)
    _qre_fredkin_stair(reg_a, reg_b)


def qre_prepare_dicke_state(wires, k):
    """qre mirror of dicke_state_prep.prepare_dicke_state."""
    n = len(wires)
    if k == 0:
        return
    if k == n:
        for w in wires:
            qre.X(wires=w)
        return
    for w in wires[:k]:
        qre.X(wires=w)
    tree = _build_partition_tree(list(wires), k)
    for v in tree.get_internal_nodes():
        _qre_apply_wdb(v, k)
    for u in tree.get_leaves():
        qubits = u.get_qubits()
        _qre_reverse(qubits)
        _qre_dicke_state_scs(len(qubits), len(qubits), qubits)


def qre_prepare_dicke_multiweight_state(wires, weights):
    """qre mirror of dicke_state_prep.prepare_dicke_multiweight_state.

    Assumes the weight set is (or is treated as) the contiguous range
    [0, max(weights)] -- true for every gadget family in this codebase
    (Dicke-LEQ, flow, cardinality-GEQ-single via X-gates handled separately).
    Under that assumption the real circuit's staircase never skips a gate
    (the "skip if computed angle ~= 0" branch only fires when a weight
    outside the set would otherwise get non-zero amplitude, which cannot
    happen for a contiguous range) -- this mirror relies on the same
    assumption the closed-form counts it replaces relied on.
    """
    n = len(wires)
    max_w = max(weights)
    if max_w == 0:
        return
    qre.RY(wires=wires[0])
    for j in range(1, max_w):
        qre.CRY(wires=[wires[j - 1], wires[j]])
    _qre_reverse(list(wires))
    _qre_dicke_state_scs(n, max_w, list(wires))


def qre_prepare_flow_state(in_wires, out_wires):
    """qre mirror of dicke_state_prep.prepare_flow_state."""
    n_in, n_out = len(in_wires), len(out_wires)
    max_w = min(n_in, n_out)
    if max_w == 0:
        return
    qre.RY(wires=in_wires[0])
    for j in range(1, max_w):
        qre.CRY(wires=[in_wires[j - 1], in_wires[j]])
    for j in range(max_w):
        qre.CNOT(wires=[in_wires[j], out_wires[j]])
    _qre_reverse(list(in_wires))
    _qre_dicke_state_scs(n_in, max_w, list(in_wires))
    _qre_reverse(list(out_wires))
    _qre_dicke_state_scs(n_out, max_w, list(out_wires))


def _qre_x_all(wires):
    for w in wires:
        qre.X(wires=w)


def _qre_hadamard_init(wires):
    for w in wires:
        qre.Hadamard(wires=w)


def _qre_x_mixer(wires):
    for w in wires:
        qre.RX(wires=w)


def _qre_xy_interaction(a, b):
    """One IsingXY(theta) interaction: standard 2-CNOT + 2-RY realisation."""
    qre.CNOT(wires=[a, b])
    qre.RY(wires=a)
    qre.CNOT(wires=[a, b])
    qre.RY(wires=b)


def _qre_xy_mixer(wires, ring=True):
    """qre mirror of qaoa_base.apply_xy_mixer."""
    n = len(wires)
    if n < 2:
        return
    pairs = [(wires[k], wires[k + 1]) for k in range(0, n - 1, 2)]
    pairs += [(wires[k], wires[k + 1]) for k in range(1, n - 1, 2)]
    if ring and n > 2:
        pairs.append((wires[-1], wires[0]))
    for a, b in pairs:
        _qre_xy_interaction(a, b)


def _qre_grover_reflection_on(wires):
    """qre mirror of the reflection portion of qaoa_base.apply_grover_mixer."""
    n = len(wires)
    if n == 0:
        return
    _qre_x_all(wires)
    for c in range(1, n):
        qre.Controlled(qre.RZ(), num_ctrl_wires=c, num_zero_ctrl=0, wires=wires[:c + 1])
    qre.PhaseShift(wires=wires[0])
    _qre_x_all(wires)


def _qre_cost_layer(hamiltonian: qml.Hamiltonian):
    """qre mirror of qaoa_base.apply_cost_unitary (angle-independent: one
    MultiRZ per non-identity Pauli term, on that term's real wires)."""
    _, ops = hamiltonian.terms()
    for op in ops:
        if _re.search(r"^I+$", qml.pauli.pauli_word_to_string(op)):
            continue
        qre.MultiRZ(num_wires=len(op.wires), wires=list(op.wires))


# ── Gadget descriptors: {"kind", "wires", "fn", ["n_in"], ["missing"]} ─────
#
# One descriptor per structural constraint, built either from a real
# gadget object (DickeStatePrep, VCG, ...) or -- for VCG -- reconstructed
# from a saved {opt_angles, n_layers, constraint_Ham} training-database
# entry.  "fn" is a zero-arg callable emitting that gadget's *state-prep*
# circuit alone (qre gates); the mixer (XY / Grover / none) is layered on
# top separately in ``_qre_mixer_layer`` since it depends on the gadget kind.

def _descriptor_from_gadget_obj(obj) -> dict:
    from . import dicke_state_prep as dsp

    if isinstance(obj, dsp.DickeStatePrep):
        return {"kind": "dicke", "wires": obj.var_wires,
                "fn": _closure(qre_prepare_dicke_state, obj.var_wires, obj.hamming_weight)}
    if isinstance(obj, dsp.CardinalityLeqStatePrep):
        k = obj.max_hamming_weight
        fn = (lambda: None) if k == 0 else _closure(
            qre_prepare_dicke_multiweight_state, obj.var_wires, list(range(k + 1)))
        return {"kind": "leq", "wires": obj.var_wires, "fn": fn}
    if isinstance(obj, dsp.CardinalityGeqSingleStatePrep):
        return {"kind": "geq", "wires": obj.var_wires, "fn": _closure(_qre_x_all, obj.var_wires)}
    if isinstance(obj, dsp.FlowStatePrep):
        wires = list(obj.in_wires) + list(obj.out_wires)
        return {"kind": "flow", "wires": wires,
                "fn": _closure(qre_prepare_flow_state, obj.in_wires, obj.out_wires),
                "n_in": obj.n_in}
    if isinstance(obj, dsp.DickeMultiweightStatePrep):
        return {"kind": "leq", "wires": obj.var_wires,
                "fn": _closure(qre_prepare_dicke_multiweight_state, obj.var_wires, obj.weights)}

    from .vcg import VCG
    if isinstance(obj, VCG):
        return _vcg_descriptor_from_object(obj)
    raise TypeError(f"Unrecognised state-prep object: {type(obj)!r}")


def _vcg_descriptor_from_object(gadget) -> dict:
    """Build a qre descriptor from a (trained, or DB-restored) VCG instance.

    Mirrors VCG.opt_circuit(): X-gate / exact-Dicke special cases, or
    Hadamard-init + n_layers x (cost-unitary MultiRZ per term + X-mixer),
    which is exactly VCG._circuit()'s ``decompose=True`` path.
    """
    wires = list(gadget.var_wires)

    if gadget._single_feasible_bitstring is not None:
        bits = [w for w, b in zip(wires, gadget._single_feasible_bitstring) if b == "1"]
        return {"kind": "vcg", "wires": wires, "fn": _closure(_qre_x_all, bits)}

    if gadget._dicke_superposition_weights is not None:
        weights = list(gadget._dicke_superposition_weights)
        return {"kind": "vcg", "wires": wires,
                "fn": _closure(qre_prepare_dicke_multiweight_state, wires, weights)}

    if gadget.opt_angles is None or not gadget.n_layers:
        return {"kind": "vcg", "wires": wires, "fn": _closure(_qre_hadamard_init, wires)}

    if not gadget.decompose:
        raise NotImplementedError(
            "VCG resource estimation only supports decompose=True "
            "(Pauli-decomposed cost unitary); this gadget used the "
            "DiagonalQubitUnitary path."
        )

    term_wires = [list(op.wires) for op in gadget.constraint_Ham.terms()[1]
                  if not _re.search(r"^I+$", qml.pauli.pauli_word_to_string(op))]
    n_layers = gadget.n_layers

    def _fn(wires=wires, term_wires=term_wires, n_layers=n_layers):
        _qre_hadamard_init(wires)
        for _ in range(n_layers):
            for tw in term_wires:
                qre.MultiRZ(num_wires=len(tw), wires=tw)
            _qre_x_mixer(wires)

    return {"kind": "vcg", "wires": wires, "fn": _fn}


def _vcg_gadget_from_db(pc, constraint_str, vcg_db) -> dict | None:
    """DB-hit branch of pc_qaoa._load_vcg_gadget, reconstructed as a
    qre descriptor.  Returns None on a cache miss (does NOT fall back to
    training -- that is a real, slow QAOA optimisation and must never be
    triggered as a side effect of resource estimation)."""
    if vcg_db is None:
        return None
    key = ch.normalize_constraint(constraint_str)
    if key not in vcg_db:
        return None
    entry = vcg_db[key]

    from .vcg import VCG
    gadget = VCG(constraints=[constraint_str])
    gadget.opt_angles = entry.get("opt_angles")
    gadget.n_layers = entry.get("n_layers")
    gadget._single_feasible_bitstring = entry.get("single_feasible_bitstring")
    gadget._dicke_superposition_weights = entry.get("dicke_superposition_weights")
    return _vcg_descriptor_from_object(gadget)


def _gadget_descriptor(pc, constraint_str, vcg_db=None) -> dict:
    """Build a gadget descriptor from a ParsedConstraint by classification
    (mirrors the dispatch in estimate_from_task / PC-QAOA construction)."""
    from . import dicke_state_prep as dsp

    if ch.is_dicke_compatible(pc):
        return _descriptor_from_gadget_obj(dsp.from_parsed_constraint(pc))
    if ch.is_cardinality_leq_compatible(pc):
        return _descriptor_from_gadget_obj(dsp.from_cardinality_leq_constraint(pc))
    if ch.is_independent_set_pair_compatible(pc):
        wires = sorted(pc.variables)
        return {"kind": "leq", "wires": wires,
                "fn": _closure(qre_prepare_dicke_multiweight_state, wires, [0, 1])}
    if ch.is_cardinality_geq_single_compatible(pc):
        return _descriptor_from_gadget_obj(dsp.from_cardinality_geq_single_constraint(pc))
    if ch.is_flow_compatible(pc):
        return _descriptor_from_gadget_obj(dsp.from_flow_constraint(pc))

    d = _vcg_gadget_from_db(pc, constraint_str, vcg_db)
    if d is None:
        return {"kind": "vcg", "wires": sorted(pc.variables), "fn": None, "missing": True}
    d["missing"] = False
    return d


# ── Full-circuit qre mirrors (state prep / one layer) ──────────────────────

def _qre_state_prep_layer(slack_wires, descriptors):
    _qre_hadamard_init(slack_wires)
    for d in descriptors:
        if d.get("fn") is not None:
            d["fn"]()


def _qre_mixer_layer(all_wires, descriptors, problem_ham):
    """One PC-QAOA layer: cost unitary + per-gadget mixer + X-mixer on free wires."""
    _qre_cost_layer(problem_ham)
    covered = set()
    for d in descriptors:
        fn = d.get("fn")
        if fn is None:  # VCG gadget missing from the training DB
            continue
        kind = d["kind"]
        covered.update(d["wires"])
        if kind == "dicke":
            _qre_xy_mixer(d["wires"], ring=True)
        elif kind == "flow":
            n_in = d["n_in"]
            wires = d["wires"]
            _qre_xy_mixer(wires[:n_in], ring=True)
            _qre_xy_mixer(wires[n_in:], ring=True)
        elif kind in ("leq", "vcg"):
            fn()
            _qre_grover_reflection_on(d["wires"])
            fn()
        # "geq": fixed by construction, no mixer.
    free_wires = [w for w in all_wires if w not in covered]
    _qre_x_mixer(free_wires)


def _qre_penalty_layer(all_wires, hamiltonian):
    _qre_cost_layer(hamiltonian)
    _qre_x_mixer(all_wires)


# ── Per-constraint resource estimation (gadget vs. penalty) ────────────────

def estimate_constraint_gadget(constraint_str: str, vcg_db: dict | None = None,
                               gate_sets=None) -> dict:
    """Resources for using ONE constraint as an exact/VCG state-prep gadget.

    Returns {"kind": ..., "missing": bool, "nisq": {...}, "ftqc": {...}}
    ("missing" is True only for a VCG constraint absent from vcg_db, in
    which case no gate-set entries are present).
    """
    gate_sets = gate_sets or GATE_SETS
    pc = ch.parse_constraints([constraint_str])[0]
    d = _gadget_descriptor(pc, constraint_str, vcg_db)
    out = {"kind": d["kind"], "missing": d.get("missing", False)}
    if out["missing"]:
        return out
    for gs_name, gate_set in gate_sets.items():
        out[gs_name] = resources_to_dict(_run_estimate(d["fn"], gate_set))
    return out


def estimate_constraint_penalty(constraint_str: str, n_x: int, penalty_weight: float,
                                gate_sets=None) -> dict:
    """Resources for penalizing ONE constraint as its own term in the cost
    Hamiltonian -- just that term (no mixer), reusing the real
    build_penalty_hamiltonian / apply_cost_unitary construction."""
    from . import qaoa_base as base

    gate_sets = gate_sets or GATE_SETS
    pc = ch.parse_constraints([constraint_str])[0]
    slack_infos, _ = ch.determine_slack_variables([pc], n_x)
    pen_ham = base.build_penalty_hamiltonian([pc], slack_infos, penalty_weight)
    fn = _closure(_qre_cost_layer, pen_ham)
    return {gs_name: resources_to_dict(_run_estimate(fn, gate_set))
            for gs_name, gate_set in gate_sets.items()}


def estimate_constraint_resources(constraint_str: str, n_x: int, penalty_weight: float,
                                  vcg_db: dict | None = None, gate_sets=None) -> dict:
    """Both views for one constraint: {"gadget": ..., "penalty": ...}."""
    return {
        "gadget": estimate_constraint_gadget(constraint_str, vcg_db, gate_sets),
        "penalty": estimate_constraint_penalty(constraint_str, n_x, penalty_weight, gate_sets),
    }


def build_constraint_resource_db(constraints: List[str], n_x: int, penalty_weight: float,
                                 vcg_db: dict | None = None, gate_sets=None) -> dict:
    """{normalized_constraint: {"gadget":..., "penalty":...}} for every
    constraint in `constraints` -- save with pickle, mirroring the
    gadgets/vcg_db.pkl convention, for reuse across tasks/problems."""
    db: Dict[str, dict] = {}
    for c in constraints:
        key = ch.normalize_constraint(c)
        if key not in db:
            db[key] = estimate_constraint_resources(c, n_x, penalty_weight, vcg_db, gate_sets)
    return db


# ── Direct-instance full-circuit estimation ────────────────────────────────

def estimate_pc_qaoa_resources(pcqaoa, gate_sets=None) -> dict:
    """Estimate circuit resources for a real (possibly trained) PC-QAOA
    instance, per gate set: {"nisq": {"sp":, "layer":, "full":}, "ftqc": {...}}.
    """
    gate_sets = gate_sets or GATE_SETS
    descriptors = [_descriptor_from_gadget_obj(sp) for sp in pcqaoa.state_prep]
    slack_wires = list(range(pcqaoa.n_total - pcqaoa.n_slack, pcqaoa.n_total))
    all_wires = list(range(pcqaoa.n_total))

    out = {}
    for gs_name, gate_set in gate_sets.items():
        sp_res = _run_estimate(_closure(_qre_state_prep_layer, slack_wires, descriptors), gate_set)
        layer_res = _run_estimate(
            _closure(_qre_mixer_layer, all_wires, descriptors, pcqaoa.problem_ham), gate_set)
        full_res = sp_res.add_series(layer_res.multiply_series(pcqaoa.n_layers))
        out[gs_name] = {"sp": sp_res, "layer": layer_res, "full": full_res}
    return out


def estimate_penalty_resources(pqaoa, gate_sets=None) -> dict:
    """Estimate circuit resources for a real PenaltyQAOA instance, per gate set."""
    gate_sets = gate_sets or GATE_SETS
    slack_wires = list(range(pqaoa.n_total - pqaoa.n_slack, pqaoa.n_total))
    all_wires = list(range(pqaoa.n_total))

    out = {}
    for gs_name, gate_set in gate_sets.items():
        sp_res = _run_estimate(_closure(_qre_hadamard_init, slack_wires), gate_set)
        layer_res = _run_estimate(_closure(_qre_penalty_layer, all_wires, pqaoa.full_Ham), gate_set)
        full_res = sp_res.add_series(layer_res.multiply_series(pqaoa.n_layers))
        out[gs_name] = {"sp": sp_res, "layer": layer_res, "full": full_res}
    return out


# ── Task-level estimation (no JAX / instantiation required) ────────────────

def _build_full_penalty_ham(Q, x_wires, parsed, penalty_weight, n_x):
    """Build full QUBO + penalty Hamiltonian for PenaltyQAOA."""
    from . import qaoa_base as base
    slack_infos, _ = ch.determine_slack_variables(parsed, n_x)
    qubo_ham = base.build_qubo_hamiltonian(Q, x_wires)
    pen_ham = base.build_penalty_hamiltonian(parsed, slack_infos, penalty_weight)
    return qubo_ham + pen_ham


def estimate_from_task(
    task: dict,
    qubos: dict,
    n_layers: int = 3,
    penalty_weight: float = None,
    gate_sets=None,
    vcg_db: dict | None = None,
) -> dict:
    """
    Estimate circuit resources directly from a task dict and QUBO lookup, for
    both NISQ and FTQC gate sets.  Does NOT instantiate PC-QAOA/PenaltyQAOA
    or require JAX.

    Returns
    -------
    dict with keys:
        "n_qubits_pc", "n_slack_pc", "n_qubits_p", "n_slack_p",
        "has_vcg_pc", "vcg_missing_pc"   -- shared, gate-set-independent
        "nisq", "ftqc"  -- each a dict with:
            "pc_qaoa_sp", "pc_qaoa_layer", "pc_qaoa"   -- qre.Resources
            "penalty_sp", "penalty_layer", "penalty"   -- qre.Resources
            "sum_of_parts_pc_qaoa"   -- {gate_name: count}, upper bound:
                each structural constraint's own gadget cost (once) + each
                PC-QAOA-penalized constraint's own penalty-term cost
                (x n_layers, since that term recurs every layer)
            "sum_of_parts_penalty"   -- {gate_name: count}, upper bound:
                every constraint's own penalty-term cost (x n_layers), summed
    The "full" pc_qaoa/penalty entries can be smaller than the sum-of-parts
    entries: term-combination (Hamiltonian addition can merge two
    constraints' Pauli terms) and transpilation both reduce gate counts
    below a naive sum of independently-estimated parts.
    """
    from . import qaoa_base as base

    gate_sets = gate_sets or GATE_SETS
    all_constraints = task["constraints"]
    n_x = task["n_x"]
    Q = qubos[n_x][task["qubo_idx"]]["Q"]

    if penalty_weight is None:
        from data.make_data import get_optimal_x
        _, _, total_min = get_optimal_x(Q, all_constraints)
        penalty_weight = float(5 + 2 * abs(total_min))

    parsed = ch.parse_constraints(all_constraints)
    structural_indices, penalty_indices = ch.partition_constraints(parsed, strategy="auto")

    # ── PC-QAOA: gadget descriptors + penalty Hamiltonian for the rest ─────
    descriptors = []
    vcg_missing = False
    for i in structural_indices:
        d = _gadget_descriptor(parsed[i], all_constraints[i], vcg_db)
        vcg_missing = vcg_missing or d.get("missing", False)
        descriptors.append(d)
    has_vcg = any(d["kind"] == "vcg" for d in descriptors)

    x_wires = list(range(n_x))
    qubo_ham = base.build_qubo_hamiltonian(Q, x_wires)
    pen_constraints = [parsed[i] for i in penalty_indices]
    if pen_constraints:
        slack_infos_pc, n_slack_pc = ch.determine_slack_variables(pen_constraints, n_x)
        pen_ham_pc = base.build_penalty_hamiltonian(pen_constraints, slack_infos_pc, penalty_weight)
        problem_ham = qubo_ham + pen_ham_pc
    else:
        n_slack_pc = 0
        problem_ham = qubo_ham
    n_total_pc = n_x + n_slack_pc
    slack_wires_pc = list(range(n_x, n_total_pc))
    all_wires_pc = list(range(n_total_pc))

    # ── Penalty-QAOA baseline: every constraint penalized ──────────────────
    pen_full_ham = _build_full_penalty_ham(Q, x_wires, parsed, penalty_weight, n_x)
    _, n_slack_p = ch.determine_slack_variables(parsed, n_x)
    n_total_p = n_x + n_slack_p
    slack_wires_p = list(range(n_x, n_total_p))
    all_wires_p = list(range(n_total_p))

    # ── Per-constraint resources (computed once, reused for both gate sets) ─
    gadget_resources = {i: estimate_constraint_gadget(all_constraints[i], vcg_db, gate_sets)
                         for i in structural_indices}
    penalty_resources = {i: estimate_constraint_penalty(all_constraints[i], n_x, penalty_weight, gate_sets)
                          for i in range(len(parsed))}

    result: Dict = {
        "n_qubits_pc": n_total_pc, "n_slack_pc": n_slack_pc,
        "n_qubits_p": n_total_p, "n_slack_p": n_slack_p,
        "has_vcg_pc": has_vcg, "vcg_missing_pc": vcg_missing,
    }

    for gs_name, gate_set in gate_sets.items():
        sp_res = _run_estimate(_closure(_qre_state_prep_layer, slack_wires_pc, descriptors), gate_set)
        layer_res = _run_estimate(
            _closure(_qre_mixer_layer, all_wires_pc, descriptors, problem_ham), gate_set)
        full_res = sp_res.add_series(layer_res.multiply_series(n_layers))

        p_sp_res = _run_estimate(_closure(_qre_hadamard_init, slack_wires_p), gate_set)
        p_layer_res = _run_estimate(_closure(_qre_penalty_layer, all_wires_p, pen_full_ham), gate_set)
        p_full_res = p_sp_res.add_series(p_layer_res.multiply_series(n_layers))

        # Gadget state-prep is applied once (like pc_qaoa_sp); a penalty term's
        # cost-unitary recurs every layer (like pc_qaoa_layer), so its count is
        # scaled by n_layers to be comparable against the full-circuit total.
        sop_pc: Dict[str, int] = {}
        for i in structural_indices:
            g = gadget_resources[i]
            if not g["missing"]:
                sop_pc = _merge_gate_dicts(sop_pc, g[gs_name]["gate_counts"])
        for i in penalty_indices:
            scaled = {k: n_layers * v for k, v in penalty_resources[i][gs_name]["gate_counts"].items()}
            sop_pc = _merge_gate_dicts(sop_pc, scaled)

        sop_penalty: Dict[str, int] = {}
        for i in range(len(parsed)):
            scaled = {k: n_layers * v for k, v in penalty_resources[i][gs_name]["gate_counts"].items()}
            sop_penalty = _merge_gate_dicts(sop_penalty, scaled)

        result[gs_name] = {
            "pc_qaoa_sp": sp_res, "pc_qaoa_layer": layer_res, "pc_qaoa": full_res,
            "penalty_sp": p_sp_res, "penalty_layer": p_layer_res, "penalty": p_full_res,
            "sum_of_parts_pc_qaoa": sop_pc,
            "sum_of_parts_penalty": sop_penalty,
        }

    return result
