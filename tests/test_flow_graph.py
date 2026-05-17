"""
test_flow_graph.py -- Test FlowStatePrep / prepare_flow_state on realistic
graph flow-conservation scenarios.

Tests:
  1. Non-contiguous wires (in/out wires interleaved on the device)
  2. Multiple independent flow constraints (different nodes in a graph)
  3. Actual named graph topologies (path, star, triangle)
  4. FlowStatePrep via from_flow_constraint (ParsedConstraint path)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from math import comb
from itertools import product as iproduct
import pennylane as qml
from pennylane import numpy as np

from core.dicke_state_prep import prepare_flow_state, FlowStatePrep
from core import constraint_handler as ch


# ---------------------------------------------------------------------------
# Generic checker
# ---------------------------------------------------------------------------

def check_flow_uniform(probs, in_wires, out_wires, all_wires, tol=1e-6, label=""):
    """
    Verify that all probability mass is on feasible states (HW(in)==HW(out))
    and that every feasible state has equal probability 1/M.
    """
    n_in  = len(in_wires)
    n_out = len(out_wires)
    max_w = min(n_in, n_out)
    M = sum(comb(n_in, w) * comb(n_out, w) for w in range(max_w + 1))
    n_total = len(all_wires)
    N = 2 ** n_total
    # wire -> bit position in probs (PennyLane: wire at index i in all_wires
    # corresponds to bit n_total-1-i in the integer index)
    wire_to_pos = {w: n_total - 1 - i for i, w in enumerate(all_wires)}

    errors = []
    for idx in range(N):
        # extract bit for each wire
        bits = {w: (idx >> wire_to_pos[w]) & 1 for w in all_wires}
        hw_in  = sum(bits[w] for w in in_wires)
        hw_out = sum(bits[w] for w in out_wires)
        # idle wires (not in in_wires or out_wires) must be 0
        idle_wires = [w for w in all_wires if w not in in_wires and w not in out_wires]
        idle_nonzero = any(bits[w] != 0 for w in idle_wires)

        p = float(probs[idx])
        if idle_nonzero or hw_in != hw_out:
            if p > tol:
                errors.append(
                    f"  idx={idx} bits={dict(sorted(bits.items()))}: "
                    f"infeasible but p={p:.6f}"
                )
        else:
            expected = 1.0 / M
            if abs(p - expected) > tol:
                errors.append(
                    f"  idx={idx} bits={dict(sorted(bits.items()))}: "
                    f"p={p:.6f}, expected {expected:.6f}"
                )

    ok = len(errors) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}  (M={M})")
    for e in errors[:5]:
        print(e)
    return ok


def run_flow_circuit(in_wires, out_wires, all_wires):
    dev = qml.device("lightning.qubit", wires=all_wires)

    @qml.qnode(dev)
    def circuit():
        prepare_flow_state(in_wires, out_wires)
        return qml.probs(wires=all_wires)

    return circuit()


def run_flow_class_circuit(in_wires, out_wires, all_wires):
    dev = qml.device("lightning.qubit", wires=all_wires)
    fsp = FlowStatePrep(in_wires=in_wires, out_wires=out_wires)

    @qml.qnode(dev)
    def circuit():
        fsp.opt_circuit()
        return qml.probs(wires=all_wires)

    return circuit()


def run_multi_flow_circuit(flow_list, all_wires):
    """Apply multiple independent flow state preps sequentially."""
    dev = qml.device("lightning.qubit", wires=all_wires)

    @qml.qnode(dev)
    def circuit():
        for in_w, out_w in flow_list:
            prepare_flow_state(in_w, out_w)
        return qml.probs(wires=all_wires)

    return circuit()


# ---------------------------------------------------------------------------
# 1. Non-contiguous wire tests
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("1. Non-contiguous wires")
print("="*60)

all_pass = True

# in and out wires interleaved: in=[0,2], out=[1,3]
in_w, out_w = [0, 2], [1, 3]
all_w = [0, 1, 2, 3]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="in=[0,2] out=[1,3]")

# in=[0,1], out=[3,4] with idle wire 2
in_w, out_w = [0, 1], [3, 4]
all_w = [0, 1, 2, 3, 4]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="in=[0,1] out=[3,4] idle=[2]")

# in=[1,3,5], out=[0,2,4] (fully reversed alternating)
in_w, out_w = [1, 3, 5], [0, 2, 4]
all_w = [0, 1, 2, 3, 4, 5]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="in=[1,3,5] out=[0,2,4]")

# asymmetric non-contiguous: in=[0,4], out=[2]
in_w, out_w = [0, 4], [2]
all_w = [0, 1, 2, 3, 4]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="in=[0,4] out=[2] idle=[1,3]")

# high-index wires: in=[5,6], out=[7,8]
in_w, out_w = [5, 6], [7, 8]
all_w = [5, 6, 7, 8]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="in=[5,6] out=[7,8] (high index)")

# class interface with non-contiguous wires
in_w, out_w = [0, 3], [1, 4]
all_w = [0, 1, 2, 3, 4]
probs = run_flow_class_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="FlowStatePrep class in=[0,3] out=[1,4] idle=[2]")


# ---------------------------------------------------------------------------
# 2. Multiple independent flow constraints (graph nodes)
# ---------------------------------------------------------------------------
# For two INDEPENDENT flow constraints (disjoint wire sets), the joint
# feasible state is the tensor product of the two individual flow states.
# Each state should be uniform over the product of feasible states.

print("\n" + "="*60)
print("2. Multiple flow constraints (graph nodes, disjoint wire sets)")
print("="*60)


def check_multi_flow(probs, flow_list, all_wires, tol=1e-6, label=""):
    """
    For disjoint independent flow constraints, check that every basis state
    which is feasible for ALL constraints has equal probability, and infeasible
    states have zero probability.
    """
    n_total = len(all_wires)
    wire_to_pos = {w: n_total - 1 - i for i, w in enumerate(all_wires)}

    # Precompute M for each constraint
    Ms = []
    for in_w, out_w in flow_list:
        n_in = len(in_w); n_out = len(out_w)
        max_w = min(n_in, n_out)
        Ms.append(sum(comb(n_in, w) * comb(n_out, w) for w in range(max_w + 1)))
    M_total = 1
    for m in Ms:
        M_total *= m

    flow_wires = set(w for fl in flow_list for in_w, out_w in [fl] for w in in_w + out_w)

    errors = []
    for idx in range(2 ** n_total):
        bits = {w: (idx >> wire_to_pos[w]) & 1 for w in all_wires}
        idle_nonzero = any(bits[w] != 0 for w in all_wires if w not in flow_wires)
        feasible = not idle_nonzero and all(
            sum(bits[w] for w in in_w) == sum(bits[w] for w in out_w)
            for in_w, out_w in flow_list
        )
        p = float(probs[idx])
        expected = 1.0 / M_total if feasible else 0.0
        if abs(p - expected) > tol:
            errors.append(
                f"  idx={idx} bits={dict(sorted(bits.items()))}: "
                f"p={p:.6f}, expected {expected:.6f}"
            )

    ok = len(errors) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}  (M_total={M_total})")
    for e in errors[:5]:
        print(e)
    return ok


# Path graph: node B has in=[edge_AB] out=[edge_BC]
# wires: edge_AB=0, edge_BC=1, edge_CD=2
# Node B: in=[0], out=[1]; Node C: in=[1], out=[2]  -- but these SHARE wire 1!
# For independent flow preps we need disjoint wire sets.
# Use: Node A out=[0,1], Node B in=[0], out=[2], Node C in=[1,2] -- still shared.
# For truly disjoint: two unrelated flow constraints on separate wire sets.

# Test A: two 1-in-1-out constraints on separate wires (disjoint)
flow_list = [([0], [1]), ([2], [3])]
all_w = [0, 1, 2, 3]
probs = run_multi_flow_circuit(flow_list, all_w)
all_pass &= check_multi_flow(probs, flow_list, all_w,
    label="2× (1-in,1-out) disjoint: {0→1} {2→3}")

# Test B: 2-in-2-out and 1-in-1-out on disjoint wires
flow_list = [([0, 1], [2, 3]), ([4], [5])]
all_w = list(range(6))
probs = run_multi_flow_circuit(flow_list, all_w)
all_pass &= check_multi_flow(probs, flow_list, all_w,
    label="(2-in,2-out) ⊗ (1-in,1-out) disjoint")

# Test C: two 2-in-2-out on disjoint wires
flow_list = [([0, 1], [2, 3]), ([4, 5], [6, 7])]
all_w = list(range(8))
probs = run_multi_flow_circuit(flow_list, all_w)
all_pass &= check_multi_flow(probs, flow_list, all_w,
    label="2× (2-in,2-out) disjoint")


# ---------------------------------------------------------------------------
# 3. Named graph topologies (single-node flow conservation)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("3. Named graph topologies (flow conservation at a single node)")
print("="*60)

# For a single flow-conservation node, we check the single FlowStatePrep.

# Star (source node): 1 in, 3 out — flow conservation at the source
in_w, out_w = [0], [1, 2, 3]
all_w = [0, 1, 2, 3]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="Star source: in=[0] out=[1,2,3]")

# Star (sink node): 3 in, 1 out — flow conservation at the sink
in_w, out_w = [0, 1, 2], [3]
all_w = [0, 1, 2, 3]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="Star sink:   in=[0,1,2] out=[3]")

# Internal node in a path graph (2 in, 2 out)
in_w, out_w = [0, 1], [2, 3]
all_w = [0, 1, 2, 3]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="Path internal: in=[0,1] out=[2,3]")

# Complete bipartite K_{2,3}: source sends 2 units over 2+3 edges
in_w, out_w = [0, 1], [2, 3, 4]
all_w = [0, 1, 2, 3, 4]
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="K_{2,3}: in=[0,1] out=[2,3,4]")

# K_{3,3}: 3 in, 3 out
in_w, out_w = [0, 1, 2], [3, 4, 5]
all_w = list(range(6))
probs = run_flow_circuit(in_w, out_w, all_w)
all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
    label="K_{3,3}: in=[0,1,2] out=[3,4,5]")


# ---------------------------------------------------------------------------
# 4. from_flow_constraint (ParsedConstraint path)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("4. from_flow_constraint (ParsedConstraint integration)")
print("="*60)

import core.dicke_state_prep as dsp

def run_from_parsed(constraint_str, all_wires_list):
    pc = ch.parse_constraints([constraint_str])[0]
    fsp = dsp.from_flow_constraint(pc)
    dev = qml.device("lightning.qubit", wires=all_wires_list)

    @qml.qnode(dev)
    def circuit():
        fsp.opt_circuit()
        return qml.probs(wires=all_wires_list)

    return fsp, circuit()

cases = [
    ("x_0 - x_1 == 0",                  [0, 1]),
    ("x_0 + x_1 - x_2 - x_3 == 0",      [0, 1, 2, 3]),
    ("x_0 + x_2 - x_1 - x_3 == 0",      [0, 1, 2, 3]),   # non-sorted coeff order
    ("x_0 + x_1 + x_2 - x_3 - x_4 == 0",[0, 1, 2, 3, 4]),
]

for cstr, all_w in cases:
    fsp, probs = run_from_parsed(cstr, all_w)
    in_w  = sorted(v for v, c in ch.parse_constraints([cstr])[0].linear.items() if c > 0)
    out_w = sorted(v for v, c in ch.parse_constraints([cstr])[0].linear.items() if c < 0)
    all_pass &= check_flow_uniform(probs, in_w, out_w, all_w,
        label=f'"{cstr}"')


# ---------------------------------------------------------------------------
print("\n" + "="*60)
print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
print("="*60)
