"""
Tests for prepare_flow_state and FlowStatePrep.opt_circuit.

For a flow constraint  Σ_in x_i - Σ_out x_j == 0  the feasible set is all
bitstrings where HW(in_wires) == HW(out_wires).  The correct uniform state is:

    (1/√M) Σ_{w=0}^{max_w} Σ_{|x|=w, |y|=w} |x⟩|y⟩

with M = Σ_w C(n_in,w)·C(n_out,w),  max_w = min(n_in, n_out).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from math import comb
import pennylane as qml
from pennylane import numpy as np
from itertools import product as iproduct

from core.dicke_state_prep import prepare_flow_state, FlowStatePrep


# ---------------------------------------------------------------------------
# Helper: enumerate feasible states and expected probabilities
# ---------------------------------------------------------------------------

def feasible_flow_states(n_in: int, n_out: int):
    """
    Return dict {bitstring: expected_probability} for the uniform flow state.
    bitstring is a tuple of length n_in + n_out (in_bits + out_bits).
    """
    max_w = min(n_in, n_out)
    M = sum(comb(n_in, w) * comb(n_out, w) for w in range(max_w + 1))
    states = {}
    for in_bits in iproduct([0, 1], repeat=n_in):
        hw_in = sum(in_bits)
        if hw_in > max_w:
            continue
        for out_bits in iproduct([0, 1], repeat=n_out):
            hw_out = sum(out_bits)
            if hw_in == hw_out:
                states[in_bits + out_bits] = 1.0 / M
    return states


def get_probs(n_in: int, n_out: int, use_class: bool = False):
    """Run the circuit and return probability array (PennyLane ordering)."""
    n_total = n_in + n_out
    in_wires = list(range(n_in))
    out_wires = list(range(n_in, n_in + n_out))

    dev = qml.device("lightning.qubit", wires=n_total)

    @qml.qnode(dev)
    def circuit():
        if use_class:
            fsp = FlowStatePrep(in_wires=in_wires, out_wires=out_wires)
            fsp.opt_circuit()
        else:
            prepare_flow_state(in_wires, out_wires)
        return qml.probs(wires=list(range(n_total)))

    return circuit()


def bits_from_index(idx: int, n_total: int):
    """Convert PennyLane state index to tuple of bits (MSB first)."""
    return tuple((idx >> (n_total - 1 - i)) & 1 for i in range(n_total))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_case(n_in: int, n_out: int, label: str, use_class: bool = False):
    probs = get_probs(n_in, n_out, use_class=use_class)
    expected = feasible_flow_states(n_in, n_out)
    n_total = n_in + n_out
    N = 2 ** n_total
    max_w = min(n_in, n_out)
    M = sum(comb(n_in, w) * comb(n_out, w) for w in range(max_w + 1))

    print(f"\n{'='*60}")
    print(f"{label}  (n_in={n_in}, n_out={n_out})")
    print(f"  max_w={max_w}, M={M}, #feasible={len(expected)}")

    errors = []
    for idx in range(N):
        bits = bits_from_index(idx, n_total)
        p = float(probs[idx])
        exp = expected.get(bits, 0.0)
        if abs(p - exp) > 1e-6:
            errors.append(f"  state {bits}: got p={p:.6f}, expected {exp:.6f}")

    if errors:
        print("  FAIL")
        for e in errors[:10]:
            print(e)
        return False
    else:
        print(f"  PASS — all {N} states match (uniform p={1/M:.6f} on feasible)")
        return True


def test_normalization(n_in: int, n_out: int, label: str):
    probs = get_probs(n_in, n_out)
    total = float(sum(probs))
    ok = abs(total - 1.0) < 1e-10
    print(f"\nNormalization {label}: total={total:.12f}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_feasibility_only(n_in: int, n_out: int, label: str):
    """All probability mass is on feasible states."""
    probs = get_probs(n_in, n_out)
    n_total = n_in + n_out
    N = 2 ** n_total
    max_w = min(n_in, n_out)

    infeasible_prob = 0.0
    for idx in range(N):
        bits = bits_from_index(idx, n_total)
        in_bits = bits[:n_in]
        out_bits = bits[n_in:]
        if sum(in_bits) != sum(out_bits):
            infeasible_prob += float(probs[idx])

    ok = infeasible_prob < 1e-10
    print(f"\nFeasibility-only {label}: infeasible_prob={infeasible_prob:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    all_pass = True

    # --- Symmetric cases (n_in == n_out) ---
    all_pass &= test_case(1, 1, "n_in=n_out=1")
    all_pass &= test_case(2, 2, "n_in=n_out=2")
    all_pass &= test_case(3, 3, "n_in=n_out=3")
    all_pass &= test_case(4, 4, "n_in=n_out=4")

    # --- Asymmetric cases ---
    all_pass &= test_case(2, 1, "n_in=2, n_out=1")
    all_pass &= test_case(1, 2, "n_in=1, n_out=2")
    all_pass &= test_case(3, 2, "n_in=3, n_out=2")
    all_pass &= test_case(4, 2, "n_in=4, n_out=2")
    all_pass &= test_case(3, 1, "n_in=3, n_out=1")

    # --- FlowStatePrep class ---
    all_pass &= test_case(2, 2, "FlowStatePrep class (2,2)", use_class=True)
    all_pass &= test_case(3, 2, "FlowStatePrep class (3,2)", use_class=True)

    # --- Normalization and feasibility ---
    for ni, no in [(2,2),(3,2),(4,3),(1,1)]:
        all_pass &= test_normalization(ni, no, f"n_in={ni},n_out={no}")
        all_pass &= test_feasibility_only(ni, no, f"n_in={ni},n_out={no}")

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
