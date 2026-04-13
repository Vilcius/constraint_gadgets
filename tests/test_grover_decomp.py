"""
test_grover_decomp.py -- Verify the ctrl(PhaseShift) decomposition.

The Grover mixer uses:
    qml.ctrl(qml.PhaseShift(beta/pi, wires=last), control=rest)

which is non-differentiable in Catalyst. We replace it with:

    ctrl(RZ(phi, last),          [c0,...,cn-2])
    ctrl(RZ(phi/2, cn-2),        [c0,...,cn-3])
    ...
    ctrl(RZ(phi/2^{n-2}, c1),    [c0])
    PhaseShift(phi/2^{n-1}, c0)

where phi = beta/pi.

This test checks that both circuits produce the same unitary matrix for
several values of n_total (2..5) and several values of phi.

Run:
    cd /home/vilcius/Papers/constraint_gadget/code
    python tests/test_grover_decomp.py
"""

import numpy as np
import pennylane as qml


def original_circuit(phi: float, all_wires: list) -> None:
    """Original Grover reflection (non-differentiable in Catalyst)."""
    for w in all_wires:
        qml.PauliX(wires=w)
    qml.ctrl(
        qml.PhaseShift(phi, wires=all_wires[-1]),
        control=all_wires[:-1],
    )
    for w in all_wires:
        qml.PauliX(wires=w)


def decomposed_circuit(phi: float, all_wires: list) -> None:
    """
    Decomposed Grover reflection using only ctrl(RZ) and PhaseShift.

    Identity used:
        ctrl(PhaseShift(phi, target), [c0,...,cn-2])
        = ctrl(RZ(phi,           target), [c0,...,cn-2])
        @ ctrl(RZ(phi/2,         cn-2),   [c0,...,cn-3])
        @ ...
        @ ctrl(RZ(phi/2^{n-2},   c1),     [c0])
        @ PhaseShift(phi/2^{n-1}, c0)

    All gates are differentiable on lightning.qubit via Catalyst adjoint.
    """
    for w in all_wires:
        qml.PauliX(wires=w)

    n = len(all_wires)
    # k=0: ctrl(RZ(phi,       last),    all controls)
    # k=1: ctrl(RZ(phi/2,     cn-2),    all controls[:-1])
    # ...
    # k=n-2: ctrl(RZ(phi/2^{n-2}, c1), [c0])
    # k=n-1: PhaseShift(phi/2^{n-1}, c0)  [no controls]
    for k in range(n - 1):
        angle = phi / (2 ** k)
        target = all_wires[-(k + 1)]         # last, then second-to-last, ...
        controls = all_wires[:n - k - 1]     # shrinking control set
        qml.ctrl(qml.RZ(angle, wires=target), control=controls)
    # Final single-qubit gate on the first wire
    qml.PhaseShift(phi / (2 ** (n - 1)), wires=all_wires[0])

    for w in all_wires:
        qml.PauliX(wires=w)


def get_unitary(circuit_fn, phi, all_wires):
    dev = qml.device("default.qubit", wires=all_wires)

    @qml.qnode(dev)
    def circ():
        circuit_fn(phi, all_wires)
        return qml.state()

    return qml.matrix(circ)()


def run_tests():
    phis = [0.0, 0.3, np.pi / 4, 1.0, np.pi, 2.3]
    n_totals = [2, 3, 4, 5]

    all_passed = True
    print(f"{'n_total':>8}  {'phi':>8}  {'max|diff|':>14}  {'pass':>6}")
    print("-" * 44)

    for n in n_totals:
        wires = list(range(n))
        for phi in phis:
            U_orig = get_unitary(original_circuit, phi, wires)
            U_new  = get_unitary(decomposed_circuit, phi, wires)
            diff = np.max(np.abs(U_orig - U_new))
            passed = diff < 1e-10
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"{n:>8}  {phi:>8.4f}  {diff:>14.2e}  {status:>6}")

    print("-" * 44)
    if all_passed:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED.")
        raise AssertionError("Matrix mismatch — decomposition is incorrect.")


if __name__ == "__main__":
    run_tests()
