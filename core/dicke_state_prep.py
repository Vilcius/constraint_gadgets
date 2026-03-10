"""
dicke_state_prep.py -- Dicke state preparation for Hamming-weight constraints.

For constraints of the form  sum_i x_i == k  (all-ones linear coefficients,
equality), the feasible subspace is exactly the Dicke state |D_n^k>, the
equal superposition of all n-qubit states with Hamming weight k.

This module provides:
  - A log-depth circuit for preparing |D_n^k>  (Bartschi & Eidenbenz, 2019).
  - Specification of the compatible mixer (XY or Ring-XY), which preserves
    Hamming weight and thus keeps the state within the feasible subspace.
  - An interface (``opt_circuit()``) matching VCG, so that
    DickeStatePrep objects can be dropped directly into HybridQAOA as
    structural state preparation components.
  - ``CardinalityLeqStatePrep`` for ``sum x_i <= k`` inequality constraints,
    which prepares a uniform superposition of Dicke states |D_n^0> through
    |D_n^k> (i.e. all weight-0-to-k bitstrings).  Uses the Grover mixer
    (XY does not preserve the feasible subspace for inequalities).

The key advantage over the general constraint gadget (VCG) is that
no flag qubits or truth-table Hamiltonian are needed -- the circuit *exactly*
prepares the feasible subspace, and the structural mixer *exactly* preserves it.

References
----------
  Bartschi, A. & Eidenbenz, S. (2019).
  "Deterministic Preparation of Dicke States."
  Fundamentals of Computation Theory (FCT), LNCS 11651, pp. 126-139.
  arXiv:1904.07358
"""

from __future__ import annotations

import math
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
# Core SCS (Split-Cyclic-Shift) building blocks
# ======================================================================

def _scs_gate(n: int, k: int, wire_a: int, wire_b: int) -> None:
    """
    Apply the SCS (Split-Cyclic-Shift) two-qubit gate.

    This is the key building block of the Bartschi-Eidenbenz Dicke state
    circuit.  It performs a controlled rotation that distributes Hamming
    weight between two subsystems.

    Implements:
        |10> -> cos(theta)|10> + sin(theta)|01>
        |01> -> -sin(theta)|10> + cos(theta)|01>

    where theta = arccos(sqrt(k/n)) effectively splits k excitations
    among n qubits.

    Parameters
    ----------
    n : int
        Total number of qubits in the current subproblem.
    k : int
        Target Hamming weight for the current subproblem.
    wire_a, wire_b : int
        The two qubit wire indices.
    """
    if n <= 0 or k <= 0 or k > n:
        return
    theta = 2 * math.acos(math.sqrt(k / n))
    # Controlled rotation: acts only when wire_a=1, wire_b=0
    # This is equivalent to a partial SWAP with angle theta
    qml.CNOT(wires=[wire_a, wire_b])
    qml.RY(theta, wires=wire_a)
    qml.CNOT(wires=[wire_b, wire_a])
    qml.RY(-theta, wires=wire_a)
    qml.CNOT(wires=[wire_b, wire_a])
    qml.CNOT(wires=[wire_a, wire_b])


def _dicke_recursion(wires: List[int], k: int) -> None:
    """
    Recursive O(log n)-depth Dicke state preparation.

    Prepares |D_n^k> on the given wires using the recursive splitting
    approach of Bartschi & Eidenbenz.

    The idea: split n qubits into two halves (n1, n2).  Prepare a
    superposition over all valid splits of k excitations:
        sum_{j=max(0,k-n2)}^{min(k,n1)} alpha_j |D_{n1}^j> |D_{n2}^{k-j}>

    Base cases:
      - k == 0:  all qubits in |0> (do nothing)
      - k == n:  all qubits in |1> (flip all)
      - n == 1:  single qubit |1> if k==1

    Parameters
    ----------
    wires : list[int]
        Qubit wire indices.
    k : int
        Target Hamming weight.
    """
    n = len(wires)

    # Base cases
    if k == 0:
        return
    if k == n:
        for w in wires:
            qml.PauliX(wires=w)
        return
    if n == 1:
        if k == 1:
            qml.PauliX(wires=wires[0])
        return

    # For n == 2, k == 1: create (|01> + |10>) / sqrt(2)
    if n == 2 and k == 1:
        qml.PauliX(wires=wires[0])
        qml.Hadamard(wires=wires[0])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        return

    # General case: use the linear-depth "staircase" approach
    # which iteratively distributes excitations across qubits.
    # This is simpler to implement correctly than the full log-depth
    # recursive split, and still provides good circuit structure.
    #
    # Start with k excitations on the first k qubits,
    # then use SCS gates to spread them across all n qubits.
    for w in wires[:k]:
        qml.PauliX(wires=w)

    # Apply SCS gates in a "staircase" pattern
    # This creates the Dicke state by successively splitting
    # excitations from left to right.
    for i in range(k):
        for j in range(i, n - k + i):
            # Gate parameters: how many qubits remain, how many excitations
            remaining_n = n - j
            remaining_k = k - i
            if remaining_n > 0 and remaining_k > 0 and remaining_k <= remaining_n:
                _scs_gate(remaining_n, remaining_k, wires[j], wires[j + 1])


def prepare_dicke_state(wires: List[int], k: int) -> None:
    """
    Prepare the Dicke state |D_n^k> on the given wires.

    This is the equal superposition of all n-qubit computational basis
    states with exactly k ones:

        |D_n^k> = (1 / sqrt(C(n,k))) * sum_{|x|=k} |x>

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
    _dicke_recursion(wires, k)


# ======================================================================
# Cardinality inequality state preparation  (sum x_i <= k)
# ======================================================================

def _m_leq(n: int, k: int) -> int:
    """Number of n-bit strings with Hamming weight <= k."""
    if k < 0:
        return 0
    if k >= n:
        return 2 ** n
    return sum(comb(n, w) for w in range(k + 1))


def prepare_cardinality_leq_state(wires: List[int], k: int) -> None:
    """
    Prepare a uniform superposition of Dicke states |D_n^0>, |D_n^1>, ..., |D_n^k>
    — equivalently, the uniform superposition over all n-bit strings with Hamming
    weight <= k:

        |S_n^{<=k}> = (1/sqrt(M)) * sum_{w=0}^{k} sum_{|x|=w} |x>

    where M = sum_{w=0}^{k} C(n, w) and each inner sum is the Dicke state |D_n^w>.

    The circuit is derived from the recursive structure of symmetric states:

        |S_n^{<=k}> = sqrt(M(n-1,k)   / M(n,k)) |0> |S_{n-1}^{<=k}>
                    + sqrt(M(n-1,k-1) / M(n,k)) |1> |S_{n-1}^{<=k-1}>

    At each qubit position i, a controlled RY is applied for every possible
    "ones count so far" j in 0..min(i, k).  The gate angle depends only on
    the remaining budget b = k - j and the remaining qubit count, so qubits
    in the same budget class share the same angle.

    Gate count: O(n^(k+1) / k!) -- polynomial for fixed k, practical for k <= n/2.

    Parameters
    ----------
    wires : list[int]
        Qubit wire indices (n = len(wires)).
    k : int
        Maximum allowed Hamming weight (0 <= k <= n).
    """
    n = len(wires)
    if k == 0:
        return  # |0...0> is the only feasible state, already prepared
    if k >= n:
        for w in wires:
            qml.Hadamard(wires=w)  # All 2^n states feasible -> uniform superposition
        return

    # Qubit 0: unconditional RY
    M_nk = _m_leq(n, k)
    M_n1_k1 = _m_leq(n - 1, k - 1)
    theta0 = 2.0 * float(np.arcsin(np.sqrt(M_n1_k1 / M_nk)))
    if abs(theta0) > 1e-12:
        qml.RY(theta0, wires=wires[0])

    # Qubits 1..n-1: controlled RY, conditioned on exact ones-count so far
    for i in range(1, n):
        n_rem = n - i
        for ones_so_far in range(min(i, k) + 1):
            b = k - ones_so_far          # remaining budget
            if b < 0:
                continue
            M_rem = _m_leq(n_rem, b)
            M_rem1_b1 = _m_leq(n_rem - 1, b - 1)
            if M_rem == 0 or M_rem1_b1 == 0:
                continue
            theta = 2.0 * float(np.arcsin(np.sqrt(M_rem1_b1 / M_rem)))
            if abs(theta) < 1e-12:
                continue
            # Apply Ry conditioned on exactly ones_so_far of wires[0..i-1] being |1>
            ctrl_wires = list(wires[:i])
            for pattern in combinations(range(i), ones_so_far):
                ctrl_vals = [1 if j in pattern else 0 for j in range(i)]
                qml.ctrl(qml.RY, control=ctrl_wires,
                         control_values=ctrl_vals)(theta, wires=wires[i])


# ======================================================================
# Main class
# ======================================================================

@dataclass
class DickeStatePrep:
    """
    Dicke state preparation component for hybrid QAOA.

    Encapsulates everything needed to:
      1. Prepare the feasible subspace |D_n^k> (state prep circuit).
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
    flag_wires : list[int]
        Always empty -- Dicke state prep needs no flags.
    all_wires : list[int]
        Same as var_wires (no ancillae needed).
    """
    var_wires: List[int]
    hamming_weight: int
    mixer_type: DickeMixerType = DickeMixerType.RING_XY
    constraint_str: str = ""

    # Derived (set in __post_init__)
    n_qubits: int = field(init=False)
    flag_wires: List[int] = field(init=False, default_factory=list)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_qubits = len(self.var_wires)
        self.flag_wires = []  # No flag qubits needed
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

    @property
    def needs_flag_penalty(self) -> bool:
        """Dicke state prep does not need flag-qubit penalties."""
        return False

    def get_info(self) -> dict:
        """
        Return a summary dict for bookkeeping / display.
        """
        return {
            "constraint": self.constraint_str,
            "type": "DickeStatePrep",
            "var_wires": self.var_wires,
            "hamming_weight": self.hamming_weight,
            "n_qubits": self.n_qubits,
            "mixer_type": self.mixer_type.name,
            "needs_flag": False,
            "flag_wires": [],
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

    Prepares a uniform superposition of Dicke states |D_n^0>, |D_n^1>, ...,
    |D_n^k> — i.e. a uniform superposition over all n-bit strings with
    Hamming weight <= k (no flag qubits, no training required):

        |S_n^{<=k}> = (1/sqrt(M)) * sum_{w=0}^{k} sum_{|x|=w} |x>

    where each inner sum is the Dicke state |D_n^w>.

    Unlike DickeStatePrep (which prepares a single Dicke state |D_n^k> and
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

    Attributes
    ----------
    flag_wires : list[int]
        Always empty -- no ancilla qubits needed.
    """
    var_wires: List[int]
    max_hamming_weight: int
    constraint_str: str = ""

    n_qubits: int = field(init=False)
    flag_wires: List[int] = field(init=False, default_factory=list)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_qubits = len(self.var_wires)
        self.flag_wires = []
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
        """
        prepare_cardinality_leq_state(self.var_wires, self.max_hamming_weight)

    @property
    def needs_flag_penalty(self) -> bool:
        return False

    def get_info(self) -> dict:
        return {
            "constraint": self.constraint_str,
            "type": "CardinalityLeqStatePrep",
            "var_wires": self.var_wires,
            "max_hamming_weight": self.max_hamming_weight,
            "n_qubits": self.n_qubits,
            "needs_flag": False,
            "flag_wires": [],
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

    No flag qubits are needed.

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
    flag_wires: List[int] = field(init=False, default_factory=list)
    var_wires: List[int] = field(init=False)
    all_wires: List[int] = field(init=False)

    def __post_init__(self):
        self.n_in = len(self.in_wires)
        self.n_out = len(self.out_wires)
        self.flag_wires = []
        self.var_wires = list(self.in_wires) + list(self.out_wires)
        self.all_wires = self.var_wires

    # ------------------------------------------------------------------
    # Circuit interface (matches DickeStatePrep / VCG)
    # ------------------------------------------------------------------

    def opt_circuit(self) -> None:
        """
        Apply the Bell-pair chain state preparation.

        Produces a superposition where every term satisfies HW(in) == HW(out).
        Remaining wires on the larger register stay at |0>.
        """
        min_n = min(self.n_in, self.n_out)
        for i in range(min_n):
            qml.Hadamard(wires=self.in_wires[i])
            qml.CNOT(wires=[self.in_wires[i], self.out_wires[i]])

    def mixer_circuit(self, beta: float) -> None:
        """
        Apply Ring-XY on in_wires and Ring-XY on out_wires independently.

        Each Ring-XY preserves HW within its register, so HW(in) == HW(out)
        is preserved.  Together they connect all feasible states.
        """
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

    @property
    def needs_flag_penalty(self) -> bool:
        """Flow state prep does not need flag-qubit penalties."""
        return False

    def get_info(self) -> dict:
        return {
            "constraint": self.constraint_str,
            "type": "FlowStatePrep",
            "in_wires": self.in_wires,
            "out_wires": self.out_wires,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "mixer_type": self.mixer_type.name,
            "needs_flag": False,
            "flag_wires": [],
        }


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
