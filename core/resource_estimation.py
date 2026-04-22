"""resource_estimation.py

Analytical circuit resource estimation for HybridQAOA and PenaltyQAOA.
Uses pennylane.labs.resource_estimation ResourceOperator subclasses.

Each ResourceOperator decomposes to primitive gate types:
    X, CNOT, Hadamard, RX, RY, RZ, CRY, CRZ, SWAP, CSWAP, PhaseShift,
    Controlled(RZ, n) for multi-controlled RZ (Grover reflection cascade).
"""
from __future__ import annotations

import re as _re
from typing import Dict, List, Tuple

import pennylane as qml
import pennylane.estimator as plre
from . import constraint_handler as ch
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

from .dicke_state_prep import _build_partition_tree


# ── Compressed-rep helpers ────────────────────────────────────────────────────

def _x():     return plre.X.resource_rep()
def _cnot():  return plre.CNOT.resource_rep()
def _h():     return plre.Hadamard.resource_rep()
def _rx():    return plre.RX.resource_rep()
def _ry():    return plre.RY.resource_rep()
def _rz():    return plre.RZ.resource_rep()
def _cry():   return plre.CRY.resource_rep()
def _crz():   return plre.CRZ.resource_rep()
def _swap():  return plre.SWAP.resource_rep()
def _cswap(): return plre.CSWAP.resource_rep()
def _ps():    return plre.PhaseShift.resource_rep()
def _mrz(n):  return plre.MultiRZ.resource_rep(num_wires=n)

def _ctrl_rz(n_ctrl: int) -> CompressedResourceOp:
    """Compressed rep for n_ctrl-controlled RZ."""
    if n_ctrl == 1:
        return _crz()
    return plre.Controlled.resource_rep(_rz(), num_ctrl_wires=n_ctrl, num_zero_ctrl=0)

def _cc_ry() -> CompressedResourceOp:
    """Compressed rep for doubly-controlled RY."""
    return plre.Controlled.resource_rep(_ry(), num_ctrl_wires=2, num_zero_ctrl=0)


# ── Dicke state gate counters (pure Python) ───────────────────────────────────

def _cnt_scs_nk(k_: int) -> Dict[str, int]:
    """Gates for _gate_scs_nk(*, k_): 2k CNOT, 1 CRY, (k-1) doubly-ctrl RY."""
    return {"CNOT": 2 * k_, "CRY": 1, "CC_RY": max(0, k_ - 1)}


def _cnt_dicke_scs(n: int, k: int) -> Dict[str, int]:
    """Gates for _dicke_state_scs(n, k) (excludes _reverse SWAPs)."""
    if k == 0:
        return {"CNOT": 0, "CRY": 0, "CC_RY": 0}
    cnot = cry = cc_ry = 0
    for _ in range(k + 1, n + 1):   # first block: (n-k) calls with weight k
        g = _cnt_scs_nk(k)
        cnot += g["CNOT"]; cry += g["CRY"]; cc_ry += g["CC_RY"]
    for l in range(2, k + 1):        # second block: (k-1) calls with weight l-1
        g = _cnt_scs_nk(l - 1)
        cnot += g["CNOT"]; cry += g["CRY"]; cc_ry += g["CC_RY"]
    return {"CNOT": cnot, "CRY": cry, "CC_RY": cc_ry}


def _cnt_fredkin_stair(ka: int, kb: int) -> Dict[str, int]:
    """Gates for _fredkin_stair with reg_a size ka, reg_b size kb."""
    cnot = cswap = 0
    const = 1
    if ka == kb:
        cnot += 1
        const = 2
    for i in range(kb - const, -1, -1):
        for _ in range(i, ka - 1):
            cswap += 1
        cnot += 1
    return {"CNOT": cnot, "CSWAP": cswap}


def _cnt_ctrl_add(ka: int, kb: int) -> Dict[str, int]:
    """Gates for _controlled_addition with reg_a size ka, reg_b size kb."""
    cry = cc_ry = 0
    for ell in range(ka - 1, -1, -1):
        lim = min(kb, ell + 1)
        for j in range(lim):
            if j == 0:
                cry += 1
            else:
                cc_ry += 1
    return {"CRY": cry, "CC_RY": cc_ry}


def _cnt_wdb(n_left: int, n_right: int, k: int) -> Dict[str, int]:
    """Gates for _apply_wdb at an internal tree node."""
    ka, kb = min(n_left, k), min(n_right, k)
    cnot = 2 * (ka - 1)         # _one_hot_encode + _revert_one_hot
    ca = _cnt_ctrl_add(ka, kb)
    fs = _cnt_fredkin_stair(ka, kb)
    return {
        "CNOT":  cnot + fs["CNOT"],
        "CRY":   ca["CRY"],
        "CC_RY": ca["CC_RY"],
        "CSWAP": fs["CSWAP"],
    }


def count_dicke_gates(n: int, k: int) -> Dict[str, int]:
    """
    Analytical gate count for prepare_dicke_state(wires, k) on n wires.

    Returns dict with keys: X, CNOT, CRY, CC_RY (doubly-controlled RY), CSWAP, SWAP.
    """
    if k == 0:
        return {"X": 0, "CNOT": 0, "CRY": 0, "CC_RY": 0, "CSWAP": 0, "SWAP": 0}
    if k == n:
        return {"X": n, "CNOT": 0, "CRY": 0, "CC_RY": 0, "CSWAP": 0, "SWAP": 0}

    counts: Dict[str, int] = {"X": k, "CNOT": 0, "CRY": 0, "CC_RY": 0, "CSWAP": 0, "SWAP": 0}
    tree = _build_partition_tree(list(range(n)), k)

    for v in tree.get_internal_nodes():
        g = _cnt_wdb(len(v.left_child.get_qubits()), len(v.right_child.get_qubits()), k)
        for key in ("CNOT", "CRY", "CC_RY", "CSWAP"):
            counts[key] += g[key]

    for u in tree.get_leaves():
        leaf_n = len(u.get_qubits())
        counts["SWAP"] += leaf_n // 2        # _reverse
        g = _cnt_dicke_scs(leaf_n, leaf_n)
        for key in ("CNOT", "CRY", "CC_RY"):
            counts[key] += g[key]

    return counts


def count_leq_gates(n: int, k: int) -> Dict[str, int]:
    """
    Analytical gate count for prepare_cardinality_leq_state(n_wires, k).

    Uses prepare_dicke_multiweight_state with weights = [0..k].
    """
    if k == 0:
        return {"RY": 0, "CRY": 0, "SWAP": 0, "CNOT": 0, "CC_RY": 0, "CSWAP": 0}
    # Step 1: staircase of k gates (j=0: RY, j=1..k-1: CRY)
    ry = 1
    cry_stair = k - 1
    # Step 2: _reverse: n//2 SWAP
    swap = n // 2
    # Step 3: _dicke_state_scs(n, k)
    g = _cnt_dicke_scs(n, k)
    return {
        "RY":    ry,
        "CRY":   cry_stair + g["CRY"],
        "SWAP":  swap,
        "CNOT":  g["CNOT"],
        "CC_RY": g["CC_RY"],
        "CSWAP": 0,
    }


def count_flow_gates(n_in: int, n_out: int) -> Dict[str, int]:
    """
    Analytical gate count for prepare_flow_state(in_wires, out_wires).
    """
    max_w = min(n_in, n_out)
    if max_w == 0:
        return {"RY": 0, "CRY": 0, "CNOT": 0, "SWAP": 0, "CC_RY": 0, "CSWAP": 0}
    # Step 1: staircase on in_wires: 1 RY + (max_w-1) CRY
    ry = 1
    cry_stair = max_w - 1
    # Step 2: CNOT copy to out_wires
    cnot_copy = max_w
    # Steps 3+4: _reverse + _dicke_state_scs on each register
    swap_in = n_in // 2
    g_in = _cnt_dicke_scs(n_in, max_w)
    swap_out = n_out // 2
    g_out = _cnt_dicke_scs(n_out, max_w)
    return {
        "RY":    ry,
        "CRY":   cry_stair + g_in["CRY"] + g_out["CRY"],
        "CNOT":  cnot_copy + g_in["CNOT"] + g_out["CNOT"],
        "SWAP":  swap_in + swap_out,
        "CC_RY": g_in["CC_RY"] + g_out["CC_RY"],
        "CSWAP": 0,
    }


# ── Hamiltonian helpers ───────────────────────────────────────────────────────

def _pauli_term_sizes(hamiltonian: qml.Hamiltonian) -> Tuple[int, ...]:
    """Wire counts for each non-identity Pauli term in the Hamiltonian."""
    sizes = []
    _, ops = hamiltonian.terms()
    for op in ops:
        s = qml.pauli.pauli_word_to_string(op)
        if not _re.search(r"^I+$", s):
            sizes.append(len(op.wires))
    return tuple(sizes)


def _state_prep_info(state_prep_list) -> Tuple:
    """
    Encode a list of state prep objects as a hashable tuple for ResourceOperator params.

    Each entry is one of:
      ("dicke", n, k)
      ("leq",   n, k)
      ("geq",   n)
      ("flow",  n_in, n_out)
      ("vcg",)          -- not analytically estimated
    """
    from . import dicke_state_prep as dsp

    entries = []
    for sp in state_prep_list:
        if isinstance(sp, dsp.DickeStatePrep):
            entries.append(("dicke", sp.n_qubits, sp.hamming_weight))
        elif isinstance(sp, dsp.CardinalityLeqStatePrep):
            entries.append(("leq", sp.n_qubits, sp.max_hamming_weight))
        elif isinstance(sp, dsp.CardinalityGeqSingleStatePrep):
            entries.append(("geq", sp.n_qubits))
        elif isinstance(sp, dsp.FlowStatePrep):
            entries.append(("flow", sp.n_in, sp.n_out))
        elif isinstance(sp, dsp.DickeMultiweightStatePrep):
            entries.append(("leq", sp.n_qubits, max(sp.weights)))
        else:
            entries.append(("vcg",))
    return tuple(entries)


def _count_state_prep(sp_info: Tuple) -> Dict[str, int]:
    """Sum gate counts over all state prep entries in sp_info."""
    total: Dict[str, int] = {}
    for entry in sp_info:
        t = entry[0]
        if t == "dicke":
            g = count_dicke_gates(entry[1], entry[2])
        elif t == "leq":
            g = count_leq_gates(entry[1], entry[2])
        elif t == "geq":
            g = {"X": entry[1]}
        elif t == "flow":
            g = count_flow_gates(entry[1], entry[2])
        else:
            g = {}   # VCG: not estimated
        for k, v in g.items():
            total[k] = total.get(k, 0) + v
    return total


# ── ResourceOperator subclasses ───────────────────────────────────────────────

class ResourceCostLayer(ResourceOperator):
    """
    Resources for one cost-unitary layer.

    Each non-identity Pauli term with k wires → MultiRZ(k) = 2(k-1) CNOT + 1 RZ.
    """

    resource_keys = {"pauli_term_sizes"}

    def __init__(self, pauli_term_sizes: Tuple[int, ...]) -> None:
        self.pauli_term_sizes = pauli_term_sizes
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"pauli_term_sizes": self.pauli_term_sizes}

    @classmethod
    def resource_rep(cls, pauli_term_sizes: Tuple[int, ...]) -> CompressedResourceOp:
        n = max(pauli_term_sizes) if pauli_term_sizes else 1
        return CompressedResourceOp(cls, n, {"pauli_term_sizes": pauli_term_sizes})

    @classmethod
    def resource_decomp(cls, pauli_term_sizes: Tuple[int, ...], **kwargs) -> List[GateCount]:
        # MultiRZ(k) = 2(k-1) CNOT + 1 RZ — group by wire count
        mrz_counts: Dict[int, int] = {}
        for k in pauli_term_sizes:
            mrz_counts[k] = mrz_counts.get(k, 0) + 1
        return [GateCount(_mrz(k), cnt) for k, cnt in mrz_counts.items()]


class ResourceXMixer(ResourceOperator):
    """Resources for one X-mixer layer: one RX per wire."""

    resource_keys = {"num_wires"}

    def __init__(self, num_wires: int) -> None:
        self.num_wires = num_wires
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires: int) -> CompressedResourceOp:
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires: int, **kwargs) -> List[GateCount]:
        return [GateCount(_rx(), num_wires)]


class ResourceXYMixer(ResourceOperator):
    """
    Resources for one XY or Ring-XY mixer layer.

    Each adjacent pair → 2 CNOT + 2 RY. Ring adds one extra pair.
    """

    resource_keys = {"num_wires", "ring"}

    def __init__(self, num_wires: int, ring: bool = True) -> None:
        self.num_wires = num_wires
        self.ring = ring
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"num_wires": self.num_wires, "ring": self.ring}

    @classmethod
    def resource_rep(cls, num_wires: int, ring: bool = True) -> CompressedResourceOp:
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires, "ring": ring})

    @classmethod
    def resource_decomp(cls, num_wires: int, ring: bool = True, **kwargs) -> List[GateCount]:
        n_pairs = num_wires if ring else (num_wires - 1)
        return [GateCount(_cnot(), 2 * n_pairs), GateCount(_ry(), 2 * n_pairs)]


class ResourceGroverReflection(ResourceOperator):
    """
    Resources for the reflection step inside one Grover-mixer layer
    (excludes state prep/unprep).

    Structure: 2n PauliX  +  ctrl(RZ, n-1) + ... + ctrl(RZ, 1)  +  1 PhaseShift  +  2n PauliX
    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires: int) -> None:
        self.num_wires = num_wires
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires: int) -> CompressedResourceOp:
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires: int, **kwargs) -> List[GateCount]:
        n = num_wires
        gates: List[GateCount] = [
            GateCount(_x(),  2 * n),    # flip all + unflip all
            GateCount(_ps(), 1),        # PhaseShift on first wire
        ]
        # ctrl(RZ) cascade: one gate for each control count c = 1 .. n-1
        for c in range(1, n):
            gates.append(GateCount(_ctrl_rz(c), 1))
        return gates


class ResourceDickeStatePrep(ResourceOperator):
    """Resources for prepare_dicke_state(n_wires, k)."""

    resource_keys = {"n", "k"}

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"n": self.n, "k": self.k}

    @classmethod
    def resource_rep(cls, n: int, k: int) -> CompressedResourceOp:
        return CompressedResourceOp(cls, n, {"n": n, "k": k})

    @classmethod
    def resource_decomp(cls, n: int, k: int, **kwargs) -> List[GateCount]:
        g = count_dicke_gates(n, k)
        rep_map = {"X": _x(), "CNOT": _cnot(), "CRY": _cry(),
                   "CC_RY": _cc_ry(), "CSWAP": _cswap(), "SWAP": _swap()}
        return [GateCount(rep_map[key], cnt) for key, cnt in g.items() if cnt > 0]


class ResourceLeqStatePrep(ResourceOperator):
    """Resources for prepare_cardinality_leq_state(n_wires, k)."""

    resource_keys = {"n", "k"}

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"n": self.n, "k": self.k}

    @classmethod
    def resource_rep(cls, n: int, k: int) -> CompressedResourceOp:
        return CompressedResourceOp(cls, n, {"n": n, "k": k})

    @classmethod
    def resource_decomp(cls, n: int, k: int, **kwargs) -> List[GateCount]:
        g = count_leq_gates(n, k)
        rep_map = {"RY": _ry(), "CRY": _cry(), "SWAP": _swap(),
                   "CNOT": _cnot(), "CC_RY": _cc_ry(), "CSWAP": _cswap()}
        return [GateCount(rep_map[key], cnt) for key, cnt in g.items() if cnt > 0]


class ResourceFlowStatePrep(ResourceOperator):
    """Resources for prepare_flow_state(in_wires, out_wires)."""

    resource_keys = {"n_in", "n_out"}

    def __init__(self, n_in: int, n_out: int) -> None:
        self.n_in = n_in
        self.n_out = n_out
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"n_in": self.n_in, "n_out": self.n_out}

    @classmethod
    def resource_rep(cls, n_in: int, n_out: int) -> CompressedResourceOp:
        return CompressedResourceOp(cls, n_in + n_out, {"n_in": n_in, "n_out": n_out})

    @classmethod
    def resource_decomp(cls, n_in: int, n_out: int, **kwargs) -> List[GateCount]:
        g = count_flow_gates(n_in, n_out)
        rep_map = {"RY": _ry(), "CRY": _cry(), "CNOT": _cnot(),
                   "SWAP": _swap(), "CC_RY": _cc_ry(), "CSWAP": _cswap()}
        return [GateCount(rep_map[key], cnt) for key, cnt in g.items() if cnt > 0]


_SP_REP_MAP = {
    "X": _x, "CNOT": _cnot, "CRY": _cry, "CC_RY": _cc_ry,
    "CSWAP": _cswap, "SWAP": _swap, "RY": _ry, "RZ": _rz,
}

# Exported gate-class sets (used by compute_vcg_resources.py and the analysis script)
_1Q_GATES_SET = frozenset({"X", "Hadamard", "RX", "RY", "RZ", "PhaseShift"})
_2Q_GATES_SET = frozenset({"CNOT", "CRY", "CRZ", "SWAP"})


class ResourceHybridStatePrep(ResourceOperator):
    """
    Initial state preparation for one HybridQAOA circuit (applied once).
    Covers: Hadamard on slack qubits + all structural state prep circuits.
    """

    resource_keys = {"n_slack", "sp_info"}

    def __init__(self, n_slack: int, sp_info: Tuple) -> None:
        self.n_slack = n_slack
        self.sp_info = sp_info
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"n_slack": self.n_slack, "sp_info": self.sp_info}

    @classmethod
    def resource_rep(cls, n_slack: int, sp_info: Tuple) -> CompressedResourceOp:
        return CompressedResourceOp(cls, 1, {"n_slack": n_slack, "sp_info": sp_info})

    @classmethod
    def resource_decomp(cls, n_slack: int, sp_info: Tuple, **kwargs) -> List[GateCount]:
        gates: List[GateCount] = []
        if n_slack > 0:
            gates.append(GateCount(_h(), n_slack))
        for key, cnt in _count_state_prep(sp_info).items():
            if cnt > 0 and key in _SP_REP_MAP:
                gates.append(GateCount(_SP_REP_MAP[key](), cnt))
        return gates


class ResourceHybridLayer(ResourceOperator):
    """
    One QAOA layer for HybridQAOA: cost unitary + mixer.

    For Grover mixer the layer cost includes 2× state prep (forward + adjoint).
    """

    resource_keys = {"num_wires", "pauli_term_sizes", "mixer", "sp_info"}

    def __init__(
        self,
        num_wires: int,
        pauli_term_sizes: Tuple[int, ...],
        mixer: str,
        sp_info: Tuple,
    ) -> None:
        self.num_wires = num_wires
        self.pauli_term_sizes = pauli_term_sizes
        self.mixer = mixer
        self.sp_info = sp_info
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {
            "num_wires": self.num_wires,
            "pauli_term_sizes": self.pauli_term_sizes,
            "mixer": self.mixer,
            "sp_info": self.sp_info,
        }

    @classmethod
    def resource_rep(cls, num_wires, pauli_term_sizes, mixer, sp_info) -> CompressedResourceOp:
        return CompressedResourceOp(cls, num_wires, {
            "num_wires": num_wires, "pauli_term_sizes": pauli_term_sizes,
            "mixer": mixer, "sp_info": sp_info,
        })

    @classmethod
    def resource_decomp(
        cls, num_wires, pauli_term_sizes, mixer, sp_info, **kwargs
    ) -> List[GateCount]:
        gates = list(ResourceCostLayer.resource_decomp(pauli_term_sizes))
        if mixer == "X-Mixer":
            gates.extend(ResourceXMixer.resource_decomp(num_wires))
        elif mixer in ("XY", "Ring-XY"):
            gates.extend(ResourceXYMixer.resource_decomp(num_wires, ring=(mixer == "Ring-XY")))
        else:  # Grover: 2× state prep + reflection
            for key, cnt in _count_state_prep(sp_info).items():
                if cnt > 0 and key in _SP_REP_MAP:
                    gates.append(GateCount(_SP_REP_MAP[key](), 2 * cnt))
            gates.extend(ResourceGroverReflection.resource_decomp(num_wires))
        return gates


class ResourcePenaltyStatePrep(ResourceOperator):
    """Initial state preparation for PenaltyQAOA: Hadamard on slack qubits only."""

    resource_keys = {"n_slack"}

    def __init__(self, n_slack: int) -> None:
        self.n_slack = n_slack
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"n_slack": self.n_slack}

    @classmethod
    def resource_rep(cls, n_slack: int) -> CompressedResourceOp:
        return CompressedResourceOp(cls, 1, {"n_slack": n_slack})

    @classmethod
    def resource_decomp(cls, n_slack: int, **kwargs) -> List[GateCount]:
        return [GateCount(_h(), n_slack)] if n_slack > 0 else []


class ResourcePenaltyLayer(ResourceOperator):
    """One QAOA layer for PenaltyQAOA: cost unitary + X mixer."""

    resource_keys = {"num_wires", "pauli_term_sizes"}

    def __init__(self, num_wires: int, pauli_term_sizes: Tuple[int, ...]) -> None:
        self.num_wires = num_wires
        self.pauli_term_sizes = pauli_term_sizes
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {"num_wires": self.num_wires, "pauli_term_sizes": self.pauli_term_sizes}

    @classmethod
    def resource_rep(cls, num_wires, pauli_term_sizes) -> CompressedResourceOp:
        return CompressedResourceOp(cls, num_wires, {
            "num_wires": num_wires, "pauli_term_sizes": pauli_term_sizes,
        })

    @classmethod
    def resource_decomp(cls, num_wires, pauli_term_sizes, **kwargs) -> List[GateCount]:
        gates = list(ResourceCostLayer.resource_decomp(pauli_term_sizes))
        gates.extend(ResourceXMixer.resource_decomp(num_wires))
        return gates


class ResourceHybridQAOA(ResourceOperator):
    """
    Full HybridQAOA circuit resources.

    Parameters
    ----------
    num_wires : int
        Total qubit count (len(all_wires)).
    n_layers : int
        Number of QAOA layers.
    pauli_term_sizes : tuple[int]
        Wire count for each non-identity Pauli term in the problem Hamiltonian.
    mixer : str
        One of "Grover", "X-Mixer", "XY", "Ring-XY".
    n_slack : int
        Number of slack qubits (each initialised with Hadamard).
    sp_info : tuple
        State prep info from _state_prep_info().  Each entry is
        ("dicke", n, k), ("leq", n, k), ("geq", n), ("flow", n_in, n_out),
        or ("vcg",).
    """

    resource_keys = {"num_wires", "n_layers", "pauli_term_sizes", "mixer", "n_slack", "sp_info"}

    def __init__(
        self,
        num_wires: int,
        n_layers: int,
        pauli_term_sizes: Tuple[int, ...],
        mixer: str,
        n_slack: int,
        sp_info: Tuple,
    ) -> None:
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.pauli_term_sizes = pauli_term_sizes
        self.mixer = mixer
        self.n_slack = n_slack
        self.sp_info = sp_info
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {
            "num_wires":       self.num_wires,
            "n_layers":        self.n_layers,
            "pauli_term_sizes": self.pauli_term_sizes,
            "mixer":           self.mixer,
            "n_slack":         self.n_slack,
            "sp_info":         self.sp_info,
        }

    @classmethod
    def resource_rep(cls, num_wires, n_layers, pauli_term_sizes, mixer, n_slack, sp_info):
        return CompressedResourceOp(cls, num_wires, {
            "num_wires": num_wires, "n_layers": n_layers,
            "pauli_term_sizes": pauli_term_sizes, "mixer": mixer,
            "n_slack": n_slack, "sp_info": sp_info,
        })

    @classmethod
    def resource_decomp(
        cls, num_wires, n_layers, pauli_term_sizes, mixer, n_slack, sp_info, **kwargs
    ) -> List[GateCount]:
        gates: List[GateCount] = []

        # 1. Slack qubit initialisation: H per slack wire
        if n_slack > 0:
            gates.append(GateCount(_h(), n_slack))

        # 2. State prep (once, before QAOA layers)
        sp_counts = _count_state_prep(sp_info)
        rep_map = {
            "X": _x(), "CNOT": _cnot(), "CRY": _cry(), "CC_RY": _cc_ry(),
            "CSWAP": _cswap(), "SWAP": _swap(), "RY": _ry(), "RZ": _rz(),
        }
        for key, cnt in sp_counts.items():
            if cnt > 0 and key in rep_map:
                gates.append(GateCount(rep_map[key], cnt))

        # 3. QAOA layers: cost unitary + mixer
        for _ in range(n_layers):
            # Cost unitary: one MultiRZ per Pauli term, grouped by wire count
            cost_layer = ResourceCostLayer.resource_decomp(pauli_term_sizes)
            gates.extend(cost_layer)

            if mixer == "X-Mixer":
                gates.extend(ResourceXMixer.resource_decomp(num_wires))
            elif mixer in ("XY", "Ring-XY"):
                ring = (mixer == "Ring-XY")
                gates.extend(ResourceXYMixer.resource_decomp(num_wires, ring=ring))
            else:  # Grover
                # 2× state prep (forward + adjoint have equal cost)
                for key, cnt in sp_counts.items():
                    if cnt > 0 and key in rep_map:
                        gates.append(GateCount(rep_map[key], 2 * cnt))
                gates.extend(ResourceGroverReflection.resource_decomp(num_wires))

        return gates


class ResourcePenaltyQAOA(ResourceOperator):
    """
    Full PenaltyQAOA circuit resources.

    Uses X mixer only. Slack qubits initialised with Hadamard.
    """

    resource_keys = {"num_wires", "n_layers", "pauli_term_sizes", "n_slack"}

    def __init__(
        self,
        num_wires: int,
        n_layers: int,
        pauli_term_sizes: Tuple[int, ...],
        n_slack: int,
    ) -> None:
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.pauli_term_sizes = pauli_term_sizes
        self.n_slack = n_slack
        super().__init__()

    @property
    def resource_params(self) -> dict:
        return {
            "num_wires":        self.num_wires,
            "n_layers":         self.n_layers,
            "pauli_term_sizes": self.pauli_term_sizes,
            "n_slack":          self.n_slack,
        }

    @classmethod
    def resource_rep(cls, num_wires, n_layers, pauli_term_sizes, n_slack):
        return CompressedResourceOp(cls, num_wires, {
            "num_wires": num_wires, "n_layers": n_layers,
            "pauli_term_sizes": pauli_term_sizes, "n_slack": n_slack,
        })

    @classmethod
    def resource_decomp(
        cls, num_wires, n_layers, pauli_term_sizes, n_slack, **kwargs
    ) -> List[GateCount]:
        gates: List[GateCount] = []

        # Slack qubit initialisation
        if n_slack > 0:
            gates.append(GateCount(_h(), n_slack))

        for _ in range(n_layers):
            gates.extend(ResourceCostLayer.resource_decomp(pauli_term_sizes))
            gates.extend(ResourceXMixer.resource_decomp(num_wires))

        return gates


# ── Top-level estimation functions ────────────────────────────────────────────

_BASE_GATE_SET = {
    "CNOT", "X", "Hadamard", "RX", "RY", "RZ", "CRY", "CRZ",
    "SWAP", "CSWAP", "PhaseShift", "MultiRZ",
}


def _build_gate_set(num_wires: int) -> set:
    """
    Build a complete gate_set for a circuit with num_wires qubits.

    Includes exact names for all Controlled gates produced by our decompositions:
      - C(RY, num_ctrl_wires=2,num_zero_ctrl=0)   [doubly-controlled RY in Dicke]
      - C(RZ, num_ctrl_wires=c,num_zero_ctrl=0)   for c=2..num_wires-1 [Grover cascade]
    """
    gate_set = set(_BASE_GATE_SET)
    gate_set.add(_cc_ry().name)
    for c in range(2, num_wires):
        gate_set.add(_ctrl_rz(c).name)
    return gate_set


def estimate_hybrid_resources(hqaoa, gate_set=None):
    """
    Estimate circuit resources for a HybridQAOA instance.
    Only reads Hamiltonian/wire metadata — does not require JIT compilation.
    """
    if gate_set is None:
        gate_set = _build_gate_set(hqaoa.n_total)
    op = ResourceHybridQAOA(
        num_wires=hqaoa.n_total,
        n_layers=hqaoa.n_layers,
        pauli_term_sizes=_pauli_term_sizes(hqaoa.problem_ham),
        mixer=hqaoa.mixer,
        n_slack=hqaoa.n_slack,
        sp_info=_state_prep_info(hqaoa.state_prep),
    )
    return plre.estimate(op, gate_set=gate_set)


def estimate_penalty_resources(pqaoa, gate_set=None):
    """
    Estimate circuit resources for a PenaltyQAOA instance.
    Only reads Hamiltonian/wire metadata — does not require JIT compilation.
    """
    if gate_set is None:
        gate_set = _build_gate_set(pqaoa.n_total)
    op = ResourcePenaltyQAOA(
        num_wires=pqaoa.n_total,
        n_layers=pqaoa.n_layers,
        pauli_term_sizes=_pauli_term_sizes(pqaoa.full_Ham),
        n_slack=pqaoa.n_slack,
    )
    return plre.estimate(op, gate_set=gate_set)


def _is_exact_structural(pc) -> bool:
    """Return True if pc can be prepared exactly (Dicke/LEQ/flow/geq-single)."""
    return (
        ch.is_dicke_compatible(pc)
        or ch.is_cardinality_leq_compatible(pc)
        or ch.is_cardinality_geq_single_compatible(pc)
        or ch.is_flow_compatible(pc)
    )


def _merge_gate_dicts(*dicts) -> dict:
    """Sum gate counts across multiple {gate_name: count} dicts."""
    merged: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def estimate_from_task(
    task: dict,
    qubos: dict,
    n_layers: int = 3,
    mixer: str = "Grover",
    penalty_weight: float = None,
    gate_set=None,
    vcg_db: dict | None = None,
) -> dict:
    """
    Estimate circuit resources directly from a task dict and QUBO lookup.

    Does NOT instantiate HybridQAOA or PenaltyQAOA — no JAX
    JIT compilation.  Builds only the Hamiltonians needed for gate counting.

    Parameters
    ----------
    task : dict
        One entry from experiment_params.jsonl.
    qubos : dict
        Output of read_qubos_from_file().
    n_layers : int
        Number of QAOA layers.
    mixer : str
        Hybrid mixer type ("Grover", "X-Mixer", "XY", "Ring-XY").
    penalty_weight : float or None
        Penalty weight.  If None, computed as 5 + 2|min_val|.
    gate_set : set[str] or None
        Gate set for plre.estimate().
    vcg_db : dict or None
        Pre-computed VCG gate counts keyed by vcg_key = str(sorted(constraints)).
        Each value is a {gate_name: count} dict for one opt_circuit() call.
        When provided, VCG structural constraints contribute to hybrid resources.

    Returns
    -------
    dict with keys:
        "hybrid", "penalty"         -- plre.Resources for full circuits
        "hybrid_sp", "hybrid_layer" -- plre.Resources for state-prep / one layer
        "penalty_sp", "penalty_layer"
        "vcg_sp_gates"              -- dict: VCG contribution to r_sp
        "vcg_layer_gates"           -- dict: VCG contribution to r_layer (2x sp)
        "has_vcg_h"                 -- bool: any VCG structural constraints?
        "vcg_missing_h"             -- bool: VCG present but not in vcg_db?
        "n_qubits_h", "n_slack_h", "n_qubits_p", "n_slack_p"
    """
    from . import qaoa_base as base
    from . import dicke_state_prep as dsp

    all_constraints = task["constraints"]
    n_x = task["n_x"]
    Q = qubos[n_x][task["qubo_idx"]]["Q"]

    if penalty_weight is None:
        from data.make_data import get_optimal_x
        _, _, total_min = get_optimal_x(Q, all_constraints)
        penalty_weight = float(5 + 2 * abs(total_min))

    parsed = ch.parse_constraints(all_constraints)
    structural_indices, penalty_indices = ch.partition_constraints(parsed, strategy="auto")
    pen_constraints = [parsed[i] for i in penalty_indices]

    # ── Hybrid wires ──────────────────────────────────────────────────────
    x_wires = list(range(n_x))
    if pen_constraints:
        _, n_slack_h = ch.determine_slack_variables(pen_constraints, n_x)
    else:
        n_slack_h = 0
    n_total_h = n_x + n_slack_h

    # Build state prep objects and classify VCG vs exact
    state_prep = []
    vcg_constraint_keys: List[str] = []   # one entry per VCG structural constraint
    for i in structural_indices:
        pc = parsed[i]
        if ch.is_dicke_compatible(pc):
            state_prep.append(dsp.from_parsed_constraint(pc))
        elif ch.is_cardinality_leq_compatible(pc):
            state_prep.append(dsp.from_cardinality_leq_constraint(pc))
        elif ch.is_cardinality_geq_single_compatible(pc):
            state_prep.append(dsp.from_cardinality_geq_single_constraint(pc))
        elif ch.is_flow_compatible(pc):
            state_prep.append(dsp.from_flow_constraint(pc))
        else:
            state_prep.append(None)  # VCG
            vcg_constraint_keys.append(str(sorted([all_constraints[i]])))

    has_vcg = len(vcg_constraint_keys) > 0
    sp_info = _state_prep_info([s for s in state_prep if s is not None])

    # ── VCG gate contributions ────────────────────────────────────────────
    vcg_sp_gates: Dict[str, int] = {}
    vcg_missing = False
    if has_vcg and vcg_db is not None:
        for key in vcg_constraint_keys:
            if key in vcg_db:
                vcg_sp_gates = _merge_gate_dicts(vcg_sp_gates, vcg_db[key])
            else:
                vcg_missing = True
    elif has_vcg:
        vcg_missing = True

    # r_layer VCG = 2× r_sp VCG (forward + adjoint in Grover)
    vcg_layer_gates: Dict[str, int] = {k: 2 * v for k, v in vcg_sp_gates.items()}

    # Build Hamiltonians (pure Python, no JAX)
    qubo_ham = base.build_qubo_hamiltonian(Q, x_wires)
    if pen_constraints:
        slack_infos, _ = ch.determine_slack_variables(pen_constraints, n_x)
        pen_ham = base.build_penalty_hamiltonian(pen_constraints, slack_infos, penalty_weight)
        problem_ham = qubo_ham + pen_ham
    else:
        problem_ham = qubo_ham

    h_term_sizes = _pauli_term_sizes(problem_ham)
    gs_h = gate_set or _build_gate_set(n_total_h)

    # ── Penalty wires ─────────────────────────────────────────────────────
    _, n_slack_p = ch.determine_slack_variables(parsed, n_x)
    n_total_p = n_x + n_slack_p
    pen_full_ham = _build_full_penalty_ham(Q, x_wires, parsed, penalty_weight, n_x)
    p_term_sizes = _pauli_term_sizes(pen_full_ham)
    gs_p = gate_set or _build_gate_set(n_total_p)

    # ── Factored estimates (non-VCG): state prep (once) + per layer ───────
    h_sp  = plre.estimate(ResourceHybridStatePrep(n_slack_h, sp_info), gate_set=gs_h)
    h_lay = plre.estimate(ResourceHybridLayer(n_total_h, h_term_sizes, mixer, sp_info),
                          gate_set=gs_h)
    p_sp  = plre.estimate(ResourcePenaltyStatePrep(n_slack_p), gate_set=gs_p)
    p_lay = plre.estimate(ResourcePenaltyLayer(n_total_p, p_term_sizes), gate_set=gs_p)

    # ── Full circuit (r_sp + n_layers * r_layer) ──────────────────────────
    hybrid_op = ResourceHybridQAOA(
        num_wires=n_total_h, n_layers=n_layers,
        pauli_term_sizes=h_term_sizes, mixer=mixer,
        n_slack=n_slack_h, sp_info=sp_info,
    )
    penalty_op = ResourcePenaltyQAOA(
        num_wires=n_total_p, n_layers=n_layers,
        pauli_term_sizes=p_term_sizes, n_slack=n_slack_p,
    )

    return {
        "hybrid":           plre.estimate(hybrid_op,  gate_set=gs_h),
        "penalty":          plre.estimate(penalty_op, gate_set=gs_p),
        "hybrid_sp":        h_sp,
        "hybrid_layer":     h_lay,
        "penalty_sp":       p_sp,
        "penalty_layer":    p_lay,
        "vcg_sp_gates":     vcg_sp_gates,
        "vcg_layer_gates":  vcg_layer_gates,
        "has_vcg_h":        has_vcg,
        "vcg_missing_h":    vcg_missing,
        "n_qubits_h":       n_total_h,
        "n_slack_h":        n_slack_h,
        "n_qubits_p":       n_total_p,
        "n_slack_p":        n_slack_p,
    }


def _build_full_penalty_ham(Q, x_wires, parsed, penalty_weight, n_x):
    """Build full QUBO + penalty Hamiltonian for PenaltyQAOA."""
    from . import qaoa_base as base
    slack_infos, _ = ch.determine_slack_variables(parsed, n_x)
    qubo_ham = base.build_qubo_hamiltonian(Q, x_wires)
    pen_ham = base.build_penalty_hamiltonian(parsed, slack_infos, penalty_weight)
    return qubo_ham + pen_ham
