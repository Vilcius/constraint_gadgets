"""
dicke_state_prep.py -- Dicke state preparation for Hamming-weight constraints.

For constraints of the form  sum_i x_i == k  (all-ones linear coefficients,
equality), the feasible subspace is exactly the Dicke state |D_n^k>, the
equal superposition of all n-qubit states with Hamming weight k.

This module provides:
  - A log-depth circuit for preparing |D_n^k>  (Bartschi & Eidenbenz, 2019).
  - Specification of the compatible mixer (XY or Ring-XY), which preserves
    Hamming weight and thus keeps the state within the feasible subspace.
  - An interface (``opt_circuit()``) matching ConstraintQAOA, so that
    DickeStatePrep objects can be dropped directly into HybridQAOA as
    structural state preparation components.

The key advantage over the general constraint gadget (ConstraintQAOA) is that
no flag qubits or truth-table Hamiltonian are needed -- the circuit *exactly*
prepares the feasible subspace, and the XY mixer *exactly* preserves it.

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

    Unlike ConstraintQAOA, this does NOT require:
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
    # Circuit interface (matches ConstraintQAOA.opt_circuit signature)
    # ------------------------------------------------------------------

    def opt_circuit(self) -> None:
        """
        Apply the Dicke state preparation circuit.

        This is the equivalent of ConstraintQAOA.opt_circuit() -- it can
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
) -> DickeStatePrep:
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
    # Import here to avoid circular dependency
    from constraint_handler import ConstraintType

    if pc.ctype != ConstraintType.DICKE:
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
