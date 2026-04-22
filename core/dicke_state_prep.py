"""
dicke_state_prep.py -- Dicke state preparation for Hamming-weight constraints.

For constraints of the form  sum_i x_i == k  (all-ones linear coefficients,
equality), the feasible subspace is exactly the Dicke state |D^n_k>, the
equal superposition of all n-qubit states with Hamming weight k.

This module provides:
  - A circuit for preparing |D^n_k>  via qubit-by-qubit conditional-RY gates.
  - Specification of the compatible mixer (XY or Ring-XY), which preserves
    Hamming weight and thus keeps the state within the feasible subspace.
  - An interface (``opt_circuit()``) matching VCG, so that
    DickeStatePrep objects can be dropped directly into HybridQAOA as
    structural state preparation components.
  - ``CardinalityLeqStatePrep`` for ``sum x_i <= k`` inequality constraints,
    which prepares a uniform superposition of Dicke states |D^n_0> through
    |D^n_k> (i.e. all weight-0-to-k bitstrings).  Uses the Grover mixer
    (XY does not preserve the feasible subspace for inequalities).

The key advantage over the general constraint gadget (VCG) is that
no truth-table Hamiltonian is needed -- the circuit *exactly*
prepares the feasible subspace, and the structural mixer *exactly* preserves it.

References
----------
  Bartschi, A. & Eidenbenz, S. (2019).
  "Deterministic Preparation of Dicke States."
  Fundamentals of Computation Theory (FCT), LNCS 11651, pp. 126-139.
  arXiv:1904.07358
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
from math import comb
from typing import List, Optional

import pennylane as qml
from pennylane import numpy as np

from . import qaoa_base as base


# ======================================================================
# Mixer types compatible with Dicke state preparation
# ======================================================================

class DickeMixerType(Enum):
    """Mixer types that preserve Hamming weight."""
    XY = auto()         # Nearest-neighbour XY interactions (open chain)
    RING_XY = auto()    # XY with periodic boundary (ring topology)
    SWAP = auto()        # Partial SWAP network


# ======================================================================
# WDB (Weight Distribution Block) + SCS (Split-Cyclic-Shift) primitives
# Ported from Bartschi & Eidenbenz (2019), arXiv:2207.09998.
# ======================================================================

# --- Partition tree ---

class _Node:
    """Binary partition tree node."""

    def __init__(self, qubits: List[int]):
        self.qubit_indices: List[int] = qubits
        self.left_child: Optional["_Node"] = None
        self.right_child: Optional["_Node"] = None

    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None

    def get_qubits(self) -> List[int]:
        return self.qubit_indices

    def get_internal_nodes(self) -> List["_Node"]:
        if self.is_leaf():
            return []
        out = [self]
        out.extend(self.left_child.get_internal_nodes())
        out.extend(self.right_child.get_internal_nodes())
        return out

    def get_leaves(self) -> List["_Node"]:
        if self.is_leaf():
            return [self]
        out = []
        out.extend(self.left_child.get_leaves())
        out.extend(self.right_child.get_leaves())
        return out


def _build_partition_tree(qubits: List[int], k: int) -> _Node:
    """Balanced binary tree with leaf size <= k."""
    if len(qubits) <= k:
        return _Node(qubits)
    chunks = [qubits[i: i + k] for i in range(0, len(qubits), k)]

    def _from_chunks(chunk_list: List[List[int]]) -> _Node:
        if len(chunk_list) == 1:
            return _Node(chunk_list[0])
        mid = len(chunk_list) // 2
        left = _from_chunks(chunk_list[:mid])
        right = _from_chunks(chunk_list[mid:])
        node = _Node(left.get_qubits() + right.get_qubits())
        node.left_child = left
        node.right_child = right
        return node

    return _from_chunks(chunks)


# --- SCS gates (leaf-level Dicke state preparation) ---

def _gate_i(n: int, wires: List[int]) -> None:
    """SCS gate type I: 2-qubit split gate."""
    qml.CNOT(wires=[wires[0], wires[1]])
    theta = 2.0 * float(np.arccos(np.sqrt(1.0 / n)))
    qml.CRY(theta, wires=[wires[1], wires[0]])
    qml.CNOT(wires=[wires[0], wires[1]])


def _gate_ii_l(l: int, n: int, wires: List[int]) -> None:
    """SCS gate type II: 3-qubit conditional split gate."""
    qml.CNOT(wires=[wires[0], wires[2]])
    theta = 2.0 * float(np.arccos(np.sqrt(float(l) / n)))
    qml.ctrl(qml.RY, control=(wires[2], wires[1]),
             control_values=(1, 1))(theta, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[2]])


def _gate_scs_nk(n: int, k: int, wires: List[int]) -> None:
    """One SCS(n,k) building block."""
    _gate_i(n, [wires[k - 1], wires[k]])
    for l in range(2, k + 1):
        _gate_ii_l(l, n, [wires[k - l], wires[k - l + 1], wires[k]])


def _scs_first_block(n: int, k: int, l: int, wires: List[int]) -> None:
    idxs = wires
    n_first = l - k - 1
    n_last = n - l
    if n_first != 0:
        idxs = idxs[n_first:]
    if n_last != 0:
        idxs = idxs[:-n_last]
    _gate_scs_nk(l, k, idxs)


def _scs_second_block(n: int, k: int, l: int, wires: List[int]) -> None:
    idxs = wires
    n_last = n - l
    if n_last != 0:
        idxs = idxs[:-n_last]
    _gate_scs_nk(l, l - 1, idxs)


def _dicke_state_scs(n: int, k: int, wires: List[int]) -> None:
    """
    SCS-based Dicke state preparation on ``wires``.

    Converts |1^k 0^{n-k}> (after _reverse) to |D^n_k>.
    Used for leaf-level preparation after WDB weight distribution.
    """
    if k == 0:
        # |D^n_0> = |0...0> — already the default computational state, no gates needed.
        return
    for l in range(k + 1, n + 1)[::-1]:
        _scs_first_block(n, k, l, wires)
    for l in range(2, k + 1)[::-1]:
        _scs_second_block(n, k, l, wires)


def _reverse(register: List[int]) -> None:
    """Reverse qubit ordering via SWAP ladder."""
    for i in range(len(register) // 2):
        qml.SWAP(wires=[register[i], register[len(register) - 1 - i]])


# --- WDB gates ---

def _compute_wdb_angles(n: int, m: int, ell: int) -> np.ndarray:
    """
    RY angles for the controlled-addition step.

    n = total qubits at node, m = right-child size, ell = weight being split.
    angle[i] = 2 * arccos( sqrt( C(m,i)*C(n-m, ell-i) / sum_{j>=i} C(m,j)*C(n-m,ell-j) ) )
    """
    x = np.array([
        comb(m, i) * comb(n - m, ell - i) if (ell - i) >= 0 else 0
        for i in range(ell + 1)
    ], dtype=float)
    s = np.array([float(np.sum(x[i:])) for i in range(ell + 1)])
    return np.where(s > 0, 2.0 * np.arccos(np.sqrt(np.clip(x / s, 0.0, 1.0))), 0.0)


def _one_hot_encode(register: List[int]) -> None:
    """Unary → one-hot via CNOT ladder (index-safe for non-contiguous wires)."""
    for i in range(len(register) - 1):
        qml.CNOT(wires=[register[i + 1], register[i]])


def _revert_one_hot(register: List[int]) -> None:
    """One-hot → unary (inverse of _one_hot_encode)."""
    for i in reversed(range(1, len(register))):
        qml.CNOT(wires=[register[i], register[i - 1]])


def _controlled_addition(
    reg_a: List[int], reg_b: List[int], n: int, m: int
) -> None:
    """Controlled-RY cascade that writes weight onto reg_b."""
    for ell in range(len(reg_a) - 1, -1, -1):
        angles = _compute_wdb_angles(n, m, ell + 1)
        for j in range(min(len(reg_b), ell + 1)):
            ctrls = [reg_a[ell]]
            if j > 0:
                ctrls = ctrls + [reg_b[j - 1]]
            qml.ctrl(qml.RY, control=ctrls,
                     control_values=[1] * len(ctrls))(float(angles[j]), wires=reg_b[j])


def _fredkin_stair(reg_a: List[int], reg_b: List[int]) -> None:
    """Fredkin staircase: subtracts distributed weight back out of reg_a."""
    const = 1
    if len(reg_a) == len(reg_b):
        qml.CNOT(wires=[reg_b[-1], reg_a[-1]])
        const = 2
    for i in range(len(reg_b) - const, -1, -1):
        for j in range(i, len(reg_a) - 1):
            qml.CSWAP(wires=[reg_b[i], reg_a[j], reg_a[j + 1]])
        qml.CNOT(wires=[reg_b[i], reg_a[-1]])


def _apply_wdb(v: _Node, k: int) -> None:
    """Apply one WDB block at internal node v."""
    reg_a = v.left_child.get_qubits()[:k]
    reg_b = v.right_child.get_qubits()[:k]
    n_v = len(v.get_qubits())
    m_v = len(v.right_child.get_qubits())   # full right-child size, not capped at k
    _one_hot_encode(reg_a)
    _controlled_addition(reg_a, reg_b, n_v, m_v)
    _revert_one_hot(reg_a)
    _fredkin_stair(reg_a, reg_b)


# ======================================================================
# Core: Dicke state preparation via WDB partition tree + SCS leaves
# ======================================================================

def prepare_dicke_state(wires: List[int], k: int) -> None:
    """
    Prepare the Dicke state |D^n_k> on the given wires.

    This is the equal superposition of all n-qubit computational basis
    states with exactly k ones:

        |D^n_k> = (1 / sqrt(C(n,k))) * sum_{|x|=k} |x>

    Implemented via the WDB (Weight Distribution Block) partition-tree
    algorithm with SCS leaf preparation (Bartschi & Eidenbenz, arXiv:2207.09998).
    Circuit depth O(k log(n/k)) vs O(n) for the naive multi-controlled-RY approach.

    Parameters
    ----------
    wires : list[int]
        Qubit wire indices (n = len(wires)).
    k : int
        Target Hamming weight (0 <= k <= n).
    """
    n = len(wires)
    if k < 0 or k > n:
        raise ValueError(f"Hamming weight k={k} out of range for n={n} qubits.")
    if k == 0:
        return  # |0...0> already prepared
    if k == n:
        for w in wires:
            qml.PauliX(wires=w)
        return

    # Initial state: |1^k 0^{n-k}>  (ones on the first k wires)
    for w in wires[:k]:
        qml.PauliX(wires=w)

    # Build partition tree with leaf size k
    tree = _build_partition_tree(list(wires), k)

    # WDB pass: distribute weight across the tree
    for v in tree.get_internal_nodes():
        _apply_wdb(v, k)

    # SCS pass: prepare uniform superposition within each leaf
    for u in tree.get_leaves():
        qubits = u.get_qubits()
        _reverse(qubits)
        _dicke_state_scs(len(qubits), len(qubits), qubits)


# ======================================================================
# Cardinality inequality state preparation  (sum x_i <= k)
# ======================================================================

def prepare_cardinality_leq_state(wires: List[int], k: int) -> None:
    """
    Prepare a uniform superposition of Dicke states |D^n_0>, |D^n_1>, ..., |D^n_k>
    — equivalently, the uniform superposition over all n-bit strings with Hamming
    weight <= k:

        |S_n^{<=k}> = (1/sqrt(M)) * sum_{w=0}^{k} sum_{|x|=w} |x>
                    = (1/sqrt(M)) * sum_{ell=0}^{k} sqrt(C(n,ell)) |D^n_ell>

    where M = sum_{w=0}^{k} C(n, w).

    Algorithm (Bartschi & Eidenbenz, arXiv:1904.07358, Theorem 2):

      Step 1 — staircase of k controlled-RY gates to build the input superposition
               sum_{ell=0}^{k} alpha_ell |1^ell 0^{n-ell}>
               where alpha_ell = sqrt(C(n, ell) / M).

               Gate j=0: unconditional RY(2 arccos(beta_0)) on wires[0].
               Gate j=1..k-1: CRY(2 arccos(beta_j)) on wires[j], controlled by
               wires[j-1] = 1, where beta_j = alpha_j / sqrt(sum_{i>=j} alpha_i^2).

      Step 2 — _reverse(wires): converts |1^ell 0^{n-ell}> -> |0^{n-ell} 1^ell>.

      Step 3 — _dicke_state_scs(n, k, wires): applies the SCS unitary U_{n,k}
               which maps each |0^{n-ell} 1^ell> -> |D^n_ell> for all ell <= k
               (Definition 2 / Lemma 2 of arXiv:1904.07358).

    Parameters
    ----------
    wires : list[int]
        Qubit wire indices (n = len(wires)).
    k : int
        Maximum allowed Hamming weight (0 <= k <= n).
    """
    prepare_dicke_multiweight_state(wires, list(range(k + 1)))


# ======================================================================
# Main class
# ======================================================================

@dataclass
class DickeStatePrep:
    """
    Dicke state preparation component for hybrid QAOA.

    Encapsulates everything needed to:
      1. Prepare the feasible subspace |D^n_k> (state prep circuit).
      2. Apply a Hamming-weight-preserving mixer.
      3. Interface with HybridQAOA via ``opt_circuit()`` and ``mixer_circuit()``.

    Unlike VCG, this does NOT require:
      - Flag qubits (the constraint is exactly satisfied by construction).
      - Truth-table / Hamiltonian decomposition.
      - Pre-training / angle optimisation for the state prep.

    Parameters
    ----------
    var_wires : list[int]
        Qubit wire indices for the decision variables in this constraint.
    hamming_weight : int
        Target Hamming weight k (from the constraint sum x_i == k).
    mixer_type : DickeMixerType
        Which Hamming-weight-preserving mixer to use.
    constraint_str : str
        Original constraint string (for bookkeeping / display).

    Attributes
    ----------
    n_qubits : int
        Number of qubits (len(var_wires)).
    all_wires : list[int]
        Same as var_wires.
    """
    var_wires: List[int]
    hamming_weight: int
    mixer_type: DickeMixerType = DickeMixerType.RING_XY
    constraint_str: str = ""

    # Derived (set in __post_init__)
    n_qubits: int = field(init=False)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_qubits = len(self.var_wires)
        self.all_wires = list(self.var_wires)

        if self.hamming_weight < 0 or self.hamming_weight > self.n_qubits:
            raise ValueError(
                f"Hamming weight {self.hamming_weight} out of range "
                f"for {self.n_qubits} qubits."
            )

    # ------------------------------------------------------------------
    # Circuit interface (matches VCG.opt_circuit signature)
    # ------------------------------------------------------------------

    def opt_circuit(self) -> None:
        """
        Apply the Dicke state preparation circuit.

        This is the equivalent of VCG.opt_circuit() -- it can
        be called by HybridQAOA as a state preparation subroutine, and
        its adjoint can be used in the Grover mixer.
        """
        prepare_dicke_state(self.var_wires, self.hamming_weight)

    def state_prep_circuit(self) -> None:
        """Alias for opt_circuit() for clarity."""
        self.opt_circuit()

    def mixer_circuit(self, beta: float) -> None:
        """
        Apply the Hamming-weight-preserving mixer.

        Parameters
        ----------
        beta : float
            Mixer angle parameter.
        """
        if self.mixer_type == DickeMixerType.XY:
            base.apply_xy_mixer(beta, self.var_wires, ring=False)
        elif self.mixer_type == DickeMixerType.RING_XY:
            base.apply_xy_mixer(beta, self.var_wires, ring=True)
        elif self.mixer_type == DickeMixerType.SWAP:
            self._apply_swap_mixer(beta)
        else:
            raise ValueError(f"Unknown mixer type: {self.mixer_type}")

    def _apply_swap_mixer(self, beta: float) -> None:
        """
        Partial-SWAP mixer: preserves Hamming weight via parameterised SWAPs.

        Applies exp(-i beta SWAP_{ij}) on neighbouring pairs, which is
        equivalent to the XY mixer up to single-qubit phases.
        """
        n = self.n_qubits
        wires = self.var_wires
        # Even bonds
        for k in range(0, n - 1, 2):
            qml.SWAP(wires=[wires[k], wires[k + 1]])
            qml.CRZ(beta, wires=[wires[k], wires[k + 1]])
            qml.SWAP(wires=[wires[k], wires[k + 1]])
        # Odd bonds
        for k in range(1, n - 1, 2):
            qml.SWAP(wires=[wires[k], wires[k + 1]])
            qml.CRZ(beta, wires=[wires[k], wires[k + 1]])
            qml.SWAP(wires=[wires[k], wires[k + 1]])

    # ------------------------------------------------------------------
    # Properties for HybridQAOA integration
    # ------------------------------------------------------------------

    @property
    def n_mixer_params(self) -> int:
        """
        Number of variational parameters for the mixer per QAOA layer.

        For global XY / Ring-XY / SWAP: 1 (single beta).
        Could be extended to per-edge parameterisation.
        """
        return 1

    def get_info(self) -> dict:
        return {
            "constraint": self.constraint_str,
            "type": "DickeStatePrep",
            "var_wires": self.var_wires,
            "hamming_weight": self.hamming_weight,
            "n_qubits": self.n_qubits,
            "mixer_type": self.mixer_type.name,
        }


# ======================================================================
# Factory: create DickeStatePrep from a ParsedConstraint
# ======================================================================

def from_parsed_constraint(
    pc,  # constraint_handler.ParsedConstraint
    mixer_type: DickeMixerType = DickeMixerType.RING_XY,
) -> "DickeStatePrep":
    """
    Create a DickeStatePrep from a ParsedConstraint.

    The constraint must be Dicke-compatible (all-ones linear, equality,
    no quadratic terms, no constant).

    Parameters
    ----------
    pc : ParsedConstraint
        A parsed constraint with ctype == ConstraintType.DICKE.
    mixer_type : DickeMixerType
        Which mixer to use.

    Returns
    -------
    DickeStatePrep

    Raises
    ------
    ValueError
        If the constraint is not Dicke-compatible.
    """
    from . import constraint_handler as ch

    if pc.ctype != ch.ConstraintType.DICKE:
        raise ValueError(
            f"Constraint '{pc.raw}' is not Dicke-compatible "
            f"(type={pc.ctype.name}). "
            "Dicke state prep requires: all-ones linear coefficients, "
            "equality operator, no quadratic terms, no constant."
        )

    var_wires = sorted(pc.variables)
    k = int(pc.rhs)

    return DickeStatePrep(
        var_wires=var_wires,
        hamming_weight=k,
        mixer_type=mixer_type,
        constraint_str=pc.raw,
    )


# ======================================================================
# Cardinality inequality class  (sum x_i <= k)
# ======================================================================

@dataclass
class CardinalityLeqStatePrep:
    """
    State preparation for ``sum x_i <= k`` cardinality inequality constraints.

    Prepares a uniform superposition of Dicke states |D^n_0>, |D^n_1>, ...,
    |D^n_k> — i.e. a uniform superposition over all n-bit strings with
    Hamming weight <= k (no training required):

        |S_n^{<=k}> = (1/sqrt(M)) * sum_{w=0}^{k} sum_{|x|=w} |x>

    where each inner sum is the Dicke state |D^n_w>.

    Unlike DickeStatePrep (which prepares a single Dicke state |D^n_k> and
    uses the XY mixer), this class prepares a superposition across all Dicke
    states from weight 0 to k.  The XY mixer does NOT preserve the feasible
    subspace (it fixes weight exactly), so HybridQAOA must use the Grover
    mixer when this gadget is present.

    Parameters
    ----------
    var_wires : list[int]
        Qubit wire indices for the decision variables.
    max_hamming_weight : int
        The k in sum x_i <= k.
    constraint_str : str
        Original constraint string (for bookkeeping).

    """
    var_wires: List[int]
    max_hamming_weight: int
    constraint_str: str = ""

    n_qubits: int = field(init=False)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_qubits = len(self.var_wires)
        self.all_wires = list(self.var_wires)
        if self.max_hamming_weight < 0 or self.max_hamming_weight > self.n_qubits:
            raise ValueError(
                f"max_hamming_weight {self.max_hamming_weight} out of range "
                f"for {self.n_qubits} qubits."
            )

    def opt_circuit(self) -> None:
        """
        Apply the cardinality-inequality state preparation circuit.

        Matches the ``opt_circuit()`` interface of VCG and DickeStatePrep so
        this object can be passed to ``apply_grover_mixer`` transparently.

        Special case: when ``max_hamming_weight == 0`` the only feasible state
        is |0...0⟩, which is the default computational basis state — no gates
        are applied.
        """
        if self.max_hamming_weight == 0:
            return
        prepare_cardinality_leq_state(self.var_wires, self.max_hamming_weight)

    def get_info(self) -> dict:
        return {
            "constraint": self.constraint_str,
            "type": "CardinalityLeqStatePrep",
            "var_wires": self.var_wires,
            "max_hamming_weight": self.max_hamming_weight,
            "n_qubits": self.n_qubits,
        }


def from_cardinality_leq_constraint(
    pc,  # constraint_handler.ParsedConstraint
) -> "CardinalityLeqStatePrep":
    """
    Create a CardinalityLeqStatePrep from a ParsedConstraint.

    The constraint must be cardinality-leq-compatible: all +1 linear
    coefficients, LEQ operator, no quadratic terms, no constant.

    Parameters
    ----------
    pc : ParsedConstraint
        A parsed constraint with ctype == ConstraintType.CARDINALITY_LEQ.

    Returns
    -------
    CardinalityLeqStatePrep

    Raises
    ------
    ValueError
        If the constraint is not cardinality-leq-compatible.
    """
    from . import constraint_handler as ch

    if not ch.is_cardinality_leq_compatible(pc):
        raise ValueError(
            f"Constraint '{pc.raw}' is not cardinality-leq-compatible "
            f"(type={pc.ctype.name}). "
            "Requires: all +1 linear coefficients, LEQ operator, no quadratic, no constant."
        )

    var_wires = sorted(pc.variables)
    k = int(pc.rhs)
    return CardinalityLeqStatePrep(
        var_wires=var_wires,
        max_hamming_weight=k,
        constraint_str=pc.raw,
    )


# ======================================================================
# Cardinality GEQ single-feasible state preparation  (sum x_i >= n)
# ======================================================================

@dataclass
class CardinalityGeqSingleStatePrep:
    """
    State preparation for the single-feasible cardinality GEQ constraint.

    Handles constraints of the form ``sum x_i >= n`` where n equals the
    number of variables — the unique feasible solution is the all-ones
    bitstring.  Preparation is a PauliX gate on every variable qubit.

    This is the GEQ analogue of ``CardinalityLeqStatePrep`` with
    ``max_hamming_weight == 0``.  No training is required.

    Parameters
    ----------
    var_wires : list[int]
        Qubit wire indices for the decision variables.
    constraint_str : str
        Original constraint string (for bookkeeping only).

    """
    var_wires: List[int]
    constraint_str: str = ""

    n_qubits: int = field(init=False)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_qubits = len(self.var_wires)
        self.all_wires = list(self.var_wires)

    def opt_circuit(self) -> None:
        """Apply PauliX to every variable qubit to prepare the all-ones state."""
        for wire in self.var_wires:
            qml.PauliX(wires=wire)

    def get_info(self) -> dict:
        return {
            "constraint": self.constraint_str,
            "type": "CardinalityGeqSingleStatePrep",
            "var_wires": self.var_wires,
            "n_qubits": self.n_qubits,
        }


def from_cardinality_geq_single_constraint(
    pc,  # constraint_handler.ParsedConstraint
) -> "CardinalityGeqSingleStatePrep":
    """
    Create a CardinalityGeqSingleStatePrep from a ParsedConstraint.

    The constraint must satisfy ``is_cardinality_geq_compatible``: all +1
    linear coefficients, GEQ operator, rhs == n_vars (single feasible solution).

    Parameters
    ----------
    pc : ParsedConstraint
        A parsed constraint with ctype == ConstraintType.CARDINALITY_GEQ.

    Returns
    -------
    CardinalityGeqSingleStatePrep

    Raises
    ------
    ValueError
        If the constraint is not cardinality-geq-compatible.
    """
    from . import constraint_handler as ch

    if not ch.is_cardinality_geq_compatible(pc):
        raise ValueError(
            f"Constraint '{pc.raw}' is not cardinality-geq-compatible "
            f"(type={pc.ctype.name}). "
            "Requires: all +1 linear coefficients, GEQ operator, rhs == n_vars."
        )

    var_wires = sorted(pc.variables)
    return CardinalityGeqSingleStatePrep(
        var_wires=var_wires,
        constraint_str=pc.raw,
    )


# ======================================================================
# Flow state preparation
# ======================================================================

@dataclass
class FlowStatePrep:
    """
    Flow conservation state preparation for hybrid QAOA.

    For constraints of the form  sum_in x_i - sum_out x_j == 0
    (all ±1 linear coefficients, equality, rhs=0), the feasible subspace
    is the set of states where HW(in_wires) == HW(out_wires).

    State preparation (O(n) depth, no ancillae):
        Apply H(in_wires[i]) + CNOT(in_wires[i], out_wires[i]) for
        i = 0 ... min(n_in, n_out) - 1.  Remaining wires (on the larger
        side) are left at |0>.  Every term in the resulting superposition
        satisfies HW(in) == HW(out).

    Mixer:
        Ring-XY on in_wires + Ring-XY on out_wires (applied separately).
        Each Ring-XY preserves the Hamming weight within its register,
        hence their combination preserves HW(in) == HW(out).
        After ≥1 QAOA layer the mixer connects all feasible states
        (every sector HW = k is fully mixed within each register).

    Parameters
    ----------
    in_wires : list[int]
        Wire indices of variables with +1 coefficient.
    out_wires : list[int]
        Wire indices of variables with -1 coefficient.
    mixer_type : DickeMixerType
        Mixer for each register (default: Ring-XY).
    constraint_str : str
        Original constraint string (for bookkeeping).
    """

    in_wires: List[int]
    out_wires: List[int]
    mixer_type: DickeMixerType = DickeMixerType.RING_XY
    constraint_str: str = ""

    # Derived (set in __post_init__)
    n_in: int = field(init=False)
    n_out: int = field(init=False)
    var_wires: List[int] = field(init=False)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_in = len(self.in_wires)
        self.n_out = len(self.out_wires)
        self.var_wires = list(self.in_wires) + list(self.out_wires)
        self.all_wires = self.var_wires

    # ------------------------------------------------------------------
    # Circuit interface (matches DickeStatePrep / VCG)
    # ------------------------------------------------------------------

    def opt_circuit(self) -> None:
        """
        Apply the flow state preparation circuit.

        Produces a *uniform* superposition over all feasible states:

            (1/√M) Σ_{w=0}^{max_w} Σ_{|x|=w, |y|=w} |x⟩|y⟩
            = Σ_w α_w |D_{n_in}^w⟩ ⊗ |D_{n_out}^w⟩

        where α_w = √(C(n_in,w)·C(n_out,w)/M) and max_w = min(n_in, n_out).
        """
        prepare_flow_state(self.in_wires, self.out_wires)

    def mixer_circuit(self, beta: float) -> None:
        """
        Apply Ring-XY on in_wires and Ring-XY on out_wires independently.

        Each Ring-XY preserves HW within its register, so HW(in) == HW(out)
        is preserved.  However, the feasible space decomposes into invariant
        sectors indexed by the common weight w, and independent Ring-XY mixers
        act only within each fixed-w sector — they cannot move amplitude
        between sectors with different w.  For full feasibility-preserving
        mixing use the Grover mixer (mixer='Grover' in HybridQAOA), which is
        the default and the mixer used in all reported experiments.
        """
        import warnings
        warnings.warn(
            "FlowStatePrep.mixer_circuit uses independent Ring-XY mixers, which "
            "cannot mix across weight sectors of the flow-conservation constraint. "
            "Use mixer='Grover' in HybridQAOA for correct feasibility-preserving mixing.",
            stacklevel=2,
        )
        if self.n_in > 1:
            base.apply_xy_mixer(beta, self.in_wires, ring=True)
        if self.n_out > 1:
            base.apply_xy_mixer(beta, self.out_wires, ring=True)

    # ------------------------------------------------------------------
    # Properties for HybridQAOA integration
    # ------------------------------------------------------------------

    @property
    def n_mixer_params(self) -> int:
        """One shared beta for both in and out Ring-XY."""
        return 1

    def get_info(self) -> dict:
        return {
            "constraint": self.constraint_str,
            "type": "FlowStatePrep",
            "in_wires": self.in_wires,
            "out_wires": self.out_wires,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "mixer_type": self.mixer_type.name,
        }


# ======================================================================
# Multi-weight Dicke state preparation  (arbitrary Hamming weight set)
# ======================================================================

def prepare_dicke_multiweight_state(wires: List[int], weights: List[int]) -> None:
    """
    Prepare a uniform superposition over all bitstrings whose Hamming weight
    is in ``weights``::

        |ψ⟩ = (1/√M) Σ_{w∈W} Σ_{|x|=w} |x⟩  =  Σ_{w∈W} α_w |D^n_w⟩
              M = Σ_{w∈W} C(n, w),   α_w = √(C(n,w)/M)

    This generalises both :func:`prepare_dicke_state` (|W|=1) and
    :func:`prepare_cardinality_leq_state` (W={0,1,...,k}).

    Algorithm (Bartschi & Eidenbenz, arXiv:1904.07358, Theorem 2):

      Step 1 — staircase of max(W) controlled-RY gates to build
               Σ_{w∈W} α_w |1^w 0^{n-w}>.
               For j ∉ W: α_j = 0 → θ_j = π (X gate), skipping that weight.
               For j ∈ W: θ_j = 2 arccos(α_j / √(Σ_{i≥j} α_i²)).

      Step 2 — _reverse(wires): |1^w 0^{n-w}> -> |0^{n-w} 1^w>.

      Step 3 — _dicke_state_scs(n, max(W), wires): SCS unitary U_{n,max(W)}
               maps each |0^{n-w} 1^w> -> |D^n_w> for all w ≤ max(W).

    Parameters
    ----------
    wires : list[int]
        Qubit wire indices (n = len(wires)).
    weights : list[int]
        Target Hamming weights (0 ≤ w ≤ n for each w).
    """
    n = len(wires)
    weight_set = frozenset(weights)
    max_w = max(weights)

    if max_w == 0:
        # Only feasible state is |00...0⟩, which is already the default state.
        return

    M = sum(comb(n, w) for w in weight_set)

    if M == 0:
        return

    # --- Step 1: staircase ---
    # α_j = sqrt(C(n,j)/M) if j in W, else 0
    # beta_j = α_j / sqrt(remaining_sq),  theta_j = 2 arccos(beta_j)
    remaining_sq = 1.0
    for j in range(max_w):
        alpha_j = float(np.sqrt(comb(n, j) / M)) if j in weight_set else 0.0
        beta = alpha_j / np.sqrt(remaining_sq) if remaining_sq > 1e-14 else 0.0
        theta = 2.0 * float(np.arccos(np.clip(beta, -1.0, 1.0)))
        if abs(theta) > 1e-12:
            if j == 0:
                qml.RY(theta, wires=wires[0])
            else:
                qml.CRY(theta, wires=[wires[j - 1], wires[j]])
        remaining_sq -= alpha_j ** 2

    # --- Step 2: reverse wire order ---
    _reverse(list(wires))

    # --- Step 3: apply U_{n,max_w} to map each |0^{n-w} 1^w> -> |D^n_w> ---
    _dicke_state_scs(n, max_w, list(wires))


@dataclass
class DickeMultiweightStatePrep:
    """
    Exact state preparation for constraints whose feasible set is a uniform
    superposition over multiple Dicke states (i.e. all bitstrings whose
    Hamming weight belongs to a given set W).

    Examples
    --------
    - ``x_0 + x_1 >= 1``  →  W = {1, 2}  (all-ones GEQ, n=2)
    - ``x_0 * x_1 == 0``  →  W = {0, 1}  (product-zero, n=2)
    - ``x_0 + x_1 + x_2 <= 2``  →  W = {0, 1, 2}  (same as CardinalityLeq)

    No QAOA training required.  Requires the Grover mixer
    (the XY mixer preserves a *single* Hamming weight, not a set).

    Parameters
    ----------
    var_wires : list[int]
        Qubit wire indices.
    weights : list[int]
        Sorted list of feasible Hamming weights.
    constraint_str : str
        Original constraint string (for bookkeeping).
    """
    var_wires: List[int]
    weights: List[int]
    constraint_str: str = ""

    n_qubits: int = field(init=False)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_qubits = len(self.var_wires)
        self.all_wires = list(self.var_wires)
        n = self.n_qubits
        for w in self.weights:
            if w < 0 or w > n:
                raise ValueError(
                    f"Weight {w} out of range for {n} qubits."
                )

    def opt_circuit(self) -> None:
        """Apply the multi-weight Dicke state preparation circuit."""
        prepare_dicke_multiweight_state(self.var_wires, self.weights)

    def get_info(self) -> dict:
        return {
            "constraint": self.constraint_str,
            "type": "DickeMultiweightStatePrep",
            "var_wires": self.var_wires,
            "weights": self.weights,
            "n_qubits": self.n_qubits,
        }


def prepare_flow_state(in_wires: List[int], out_wires: List[int]) -> None:
    """
    Prepare the uniform superposition over all flow-feasible states:

        |ψ⟩ = (1/√M) Σ_{w=0}^{max_w} Σ_{|x|=w, |y|=w} |x⟩|y⟩
             = Σ_{w=0}^{max_w} α_w |D_{n_in}^w⟩ ⊗ |D_{n_out}^w⟩

    where max_w = min(n_in, n_out), α_w = √(C(n_in,w)·C(n_out,w)/M),
    and M = Σ_{w=0}^{max_w} C(n_in,w)·C(n_out,w).

    Algorithm (no ancillae, O(n) depth):

      Step 1 — Staircase of max_w controlled-RY gates on in_wires to build
               Σ_w α_w |1^w 0^{n_in-w}⟩_in.  Gate j=0 is unconditional RY
               on in_wires[0]; gates j=1,...,max_w-1 are CRY on in_wires[j]
               controlled by in_wires[j-1]=1.

      Step 2 — CNOT copy: CNOT(in_wires[j], out_wires[j]) for j=0,...,max_w-1.
               Entangles out_wires to match the weight of in_wires:
               Σ_w α_w |1^w 0^{n_in-w}⟩_in |1^w 0^{n_out-w}⟩_out.

      Step 3 — _reverse + _dicke_state_scs(n_in, max_w) on in_wires:
               U_{n_in,max_w} maps each |0^{n_in-w} 1^w⟩ → |D_{n_in}^w⟩.

      Step 4 — _reverse + _dicke_state_scs(n_out, max_w) on out_wires:
               U_{n_out,max_w} maps each |0^{n_out-w} 1^w⟩ → |D_{n_out}^w⟩.

    Steps 3 and 4 commute (different registers), giving
    Σ_w α_w |D_{n_in}^w⟩ ⊗ |D_{n_out}^w⟩.

    Parameters
    ----------
    in_wires : list[int]
        Wire indices of variables with +1 coefficient (n_in = len).
    out_wires : list[int]
        Wire indices of variables with -1 coefficient (n_out = len).
    """
    n_in = len(in_wires)
    n_out = len(out_wires)
    max_w = min(n_in, n_out)

    if max_w == 0:
        return  # trivially |0...0⟩ on both sides

    M = sum(comb(n_in, w) * comb(n_out, w) for w in range(max_w + 1))
    if M == 0:
        return

    # --- Step 1: staircase on in_wires ---
    # α_w = sqrt(C(n_in,w)*C(n_out,w)/M); loop covers w=0,...,max_w-1
    # (weight max_w gets the remaining amplitude implicitly)
    remaining_sq = 1.0
    for j in range(max_w):
        alpha_j = float(np.sqrt(comb(n_in, j) * comb(n_out, j) / M))
        beta = alpha_j / np.sqrt(remaining_sq) if remaining_sq > 1e-14 else 0.0
        theta = 2.0 * float(np.arccos(np.clip(beta, -1.0, 1.0)))
        if abs(theta) > 1e-12:
            if j == 0:
                qml.RY(theta, wires=in_wires[0])
            else:
                qml.CRY(theta, wires=[in_wires[j - 1], in_wires[j]])
        remaining_sq -= alpha_j ** 2

    # --- Step 2: CNOT copy to out_wires ---
    for j in range(max_w):
        qml.CNOT(wires=[in_wires[j], out_wires[j]])

    # --- Steps 3 & 4: U_{n_in,max_w} on in_wires, U_{n_out,max_w} on out_wires ---
    _reverse(list(in_wires))
    _dicke_state_scs(n_in, max_w, list(in_wires))
    _reverse(list(out_wires))
    _dicke_state_scs(n_out, max_w, list(out_wires))


def from_flow_constraint(
    pc,  # constraint_handler.ParsedConstraint
    mixer_type: DickeMixerType = DickeMixerType.RING_XY,
) -> FlowStatePrep:
    """
    Create a FlowStatePrep from a ParsedConstraint.

    The constraint must be flow-compatible: all ±1 linear coefficients,
    both signs present, equality, rhs=0, no quadratic terms, no constant.

    Parameters
    ----------
    pc : ParsedConstraint
        A parsed constraint with ctype == ConstraintType.FLOW.
    mixer_type : DickeMixerType
        Mixer type for both registers.

    Returns
    -------
    FlowStatePrep

    Raises
    ------
    ValueError
        If the constraint is not flow-compatible.
    """
    from . import constraint_handler as ch

    if not ch.is_flow_compatible(pc):
        raise ValueError(
            f"Constraint '{pc.raw}' is not flow-compatible "
            f"(type={pc.ctype.name}). "
            "Flow state prep requires: all ±1 linear coefficients, "
            "both signs present, equality, rhs=0, no quadratic, no constant."
        )

    in_wires = sorted(v for v, coeff in pc.linear.items() if coeff > 0)
    out_wires = sorted(v for v, coeff in pc.linear.items() if coeff < 0)

    return FlowStatePrep(
        in_wires=in_wires,
        out_wires=out_wires,
        mixer_type=mixer_type,
        constraint_str=pc.raw,
    )
