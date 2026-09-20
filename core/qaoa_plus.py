"""
qaoa_plus.py -- Literature QAOA+ baseline with JAX-jitted optimisation.

A from-scratch implementation of the Quantum Alternating Operator Ansatz
(QAOA+) as described in Hadfield et al. (Hadfield_2019): a single feasible
computational-basis initial state, and a single mixer built over the JOINT
feasible subspace of *all* structural constraints combined -- not composed
per-constraint from separate gadgets, unlike PC-QAOA (see pc_qaoa.py).

Mixer construction
-------------------
Following the general subspace-mixer recipe of Hadfield et al. and its
CNOT-efficient formalisations (fuchs2022constrained, fuchs2024lxmixers),
and generalising the hand-derived, per-problem controlled mixers of
Saleem (saleem2020) and Wang et al. (wang2023mec) past a single worked
example, the mixer is built from two families of feasibility-preserving
"moves", searched over automatically from the brute-forced feasible set F
(n_x <= ~20):

  1. Single-qubit flips (controlled-RX): toggling one qubit, controlled on
     the other n_x-1 qubits.  This is the generalised form of Saleem's
     and Wang et al.'s controlled-X independent-set mixer.  On its own it
     cannot reconnect a fixed-Hamming-weight sector, since flipping one
     bit always changes the weight by +-1.

  2. Pairwise swaps (controlled-IsingXY): swapping two qubits (01 <-> 10),
     controlled on the other n_x-2 qubits.  This is the generalised,
     automatically-derived form of the XY-mixer (wang2020xy), needed to
     reconnect Dicke-type (fixed-Hamming-weight) feasible sets, which a
     single-qubit flip alone cannot do.

For each candidate move, the controlling condition -- as a function of the
qubits *not* touched by the move -- is derived as a 3-valued Boolean
function: TRUE (both endpoints of the move are in F, so applying it is
safe), FALSE (exactly one endpoint is in F, so applying it would leave F),
or DON'T-CARE (neither endpoint is in F, so this configuration is never
reached while feasible).  This function is minimised via two-level logic
minimisation (sympy's Quine-McCluskey-based SOPform) to the smallest
number of AND-clauses, i.e. the fewest, narrowest multi-controlled gates
needed to realise the move.

As with every literature construction of this kind (Hadfield_2019,
saleem2020, wang2023mec), the resulting per-move mixer terms do not all
commute, so a single QAOA layer is a Trotterized product of them and only
approximately preserves F for finite beta -- this is inherent to the
method, not an implementation shortcut.  ``is_mixer_connected()`` reports
whether the synthesised move set actually connects all of F; the paper
already notes (Sec. sec:qaoa-plus) that an under-connected mixer can
confine QAOA+ to a suboptimal region.

Requirements:
  pip install jax optax pennylane sympy
"""

from __future__ import annotations

import itertools as it
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from sympy import symbols
from sympy.logic.boolalg import And, Or, Not, SOPform, true, false

from . import qaoa_base as base
from . import constraint_handler as ch


# ======================================================================
# Feasible-set enumeration
# ======================================================================

def _bitstrings(n: int) -> List[str]:
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def compute_feasible_states(constraints: List[str], n_x: int) -> List[str]:
    """Brute-force the joint feasible set F over ALL constraints combined."""
    parsed = ch.parse_constraints(constraints)
    return [bits for bits in _bitstrings(n_x)
            if ch.check_feasibility(bits, parsed, n_x)]


# ======================================================================
# Mixer synthesis: minimal controlled-move clauses
# ======================================================================

Clause = Tuple[List[int], List[int]]  # (control_wires, control_values)


def _minimize_to_clauses(
    syms: tuple, others: List[int], minterms: List[list], dontcares: List[list]
) -> List[Clause]:
    """Turn a minimised SOP Boolean expression into control-gate clauses."""
    if not minterms:
        return []

    expr = SOPform(list(syms), minterms, dontcares)

    if expr == true:
        return [([], [])]
    if expr == false:
        return []

    terms = expr.args if isinstance(expr, Or) else [expr]
    clauses: List[Clause] = []
    for term in terms:
        literals = term.args if isinstance(term, And) else [term]
        wires, values = [], []
        for lit in literals:
            if isinstance(lit, Not):
                var, val = lit.args[0], 0
            else:
                var, val = lit, 1
            wires.append(others[syms.index(var)])
            values.append(val)
        clauses.append((wires, values))
    return clauses


def _synthesize_move(
    feasible_set: set, n_x: int,
    target_wires: List[int], value_a: tuple, value_b: tuple,
) -> List[Clause]:
    """
    Minimal controlled-gate decomposition for toggling ``target_wires``
    between the two configurations ``value_a`` and ``value_b``, as a
    function of the other n_x - len(target_wires) qubits.
    """
    others = [w for w in range(n_x) if w not in target_wires]

    def _state(ctx_wires, ctx_bits, target_bits):
        state = [""] * n_x
        for w, b in zip(ctx_wires, ctx_bits):
            state[w] = str(b)
        for w, b in zip(target_wires, target_bits):
            state[w] = str(b)
        return "".join(state)

    if not others:
        sA = _state([], (), value_a)
        sB = _state([], (), value_b)
        return [([], [])] if sA in feasible_set and sB in feasible_set else []

    syms = symbols(f"m0:{len(others)}")
    minterms, dontcares = [], []

    for ctx_bits in it.product((0, 1), repeat=len(others)):
        sA = _state(others, ctx_bits, value_a)
        sB = _state(others, ctx_bits, value_b)
        fA, fB = sA in feasible_set, sB in feasible_set

        if fA and fB:
            minterms.append(list(ctx_bits))
        elif not fA and not fB:
            dontcares.append(list(ctx_bits))
        # exactly one of {fA, fB} feasible -> hard FALSE, excluded from both

    return _minimize_to_clauses(syms, others, minterms, dontcares)


Move = Dict  # {"kind": "flip" | "swap", "wires": [...], "clauses": [...]}


def build_mixer_moves(feasible_states: List[str], n_x: int) -> List[Move]:
    """
    Synthesise the mixer as a list of moves, each with its own minimal
    controlling clauses:

      - one single-qubit flip per qubit (controlled-RX)
      - one pairwise swap per qubit pair, complete graph (controlled-IsingXY)

    Moves with no valid (i.e. always-unsafe) clause are dropped entirely
    -- they would contribute a dead circuit parameter.
    """
    feasible_set = set(feasible_states)
    moves: List[Move] = []

    for i in range(n_x):
        clauses = _synthesize_move(feasible_set, n_x, [i], (0,), (1,))
        if clauses:
            moves.append({"kind": "flip", "wires": [i], "clauses": clauses})

    for i, j in it.combinations(range(n_x), 2):
        clauses = _synthesize_move(feasible_set, n_x, [i, j], (0, 1), (1, 0))
        if clauses:
            moves.append({"kind": "swap", "wires": [i, j], "clauses": clauses})

    return moves


# ======================================================================
# Main class
# ======================================================================

class QAOAPlus:
    """
    Literature QAOA+: single feasible initial state + a mixer built once
    over the joint feasible subspace of all constraints, via minimal
    controlled-flip and controlled-swap clauses (see module docstring).
    """

    def __init__(
        self,
        qubo,
        constraints: List[str],
        angle_strategy: str = "ma-QAOA",
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 10,
    ) -> None:
        self.angle_strategy = base.validate_angle_strategy(angle_strategy)

        self.qubo = qubo
        self.constraints = constraints
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts

        self.parsed_constraints = ch.parse_constraints(constraints)
        self.n_x = qubo.shape[0]
        self.x_wires = list(range(self.n_x))
        self.all_wires = self.x_wires
        self.n_total = self.n_x

        # --- joint feasible subspace + single feasible initial state ---
        self.feasible_states = compute_feasible_states(constraints, self.n_x)
        if not self.feasible_states:
            raise ValueError("QAOA+ requires a nonempty joint feasible set.")
        self.initial_state = min(self.feasible_states)

        # --- mixer: minimal controlled-flip / controlled-swap moves ---
        self.mixer_moves = build_mixer_moves(self.feasible_states, self.n_x)

        # --- cost Hamiltonian: plain QUBO, no penalty terms ---
        self.problem_ham = base.build_qubo_hamiltonian(self.qubo, self.x_wires)
        self.num_gamma = base.count_gamma_terms(self.problem_ham)
        self.num_beta = len(self.mixer_moves)  # one beta per active move, per layer

        self._compiled_cost, self._compiled_step = self._build_compiled_fns()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mixer_gate_count(self) -> int:
        """Total number of (possibly multi-controlled) mixer gates per layer."""
        return sum(len(move["clauses"]) for move in self.mixer_moves)

    def frozen_qubits(self) -> List[int]:
        """Qubits touched by no active move (neither flip nor swap)."""
        touched = set()
        for move in self.mixer_moves:
            touched.update(move["wires"])
        return [q for q in range(self.n_x) if q not in touched]

    def is_mixer_connected(self) -> bool:
        """
        Whether the synthesised move set connects all of F into one
        component.  A disconnected mixer confines QAOA+ to a subset of F
        (see Sec. sec:qaoa-plus's connectivity caveat).
        """
        if len(self.feasible_states) <= 1:
            return True
        feasible_set = set(self.feasible_states)
        adjacency = {x: set() for x in self.feasible_states}

        for move in self.mixer_moves:
            wires = move["wires"]
            for control_wires, control_values in move["clauses"]:
                for x in self.feasible_states:
                    if all(x[w] == str(v) for w, v in zip(control_wires, control_values)):
                        y = list(x)
                        if move["kind"] == "flip":
                            (w,) = wires
                            y[w] = "1" if x[w] == "0" else "0"
                        else:
                            i, j = wires
                            y[i], y[j] = x[j], x[i]
                        y = "".join(y)
                        if y in feasible_set:
                            adjacency[x].add(y)
                            adjacency[y].add(x)

        start = self.feasible_states[0]
        seen = {start}
        frontier = [start]
        while frontier:
            nxt = [v for u in frontier for v in adjacency[u] if v not in seen]
            seen.update(nxt)
            frontier = nxt
        return len(seen) == len(feasible_set)

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------

    def circuit(self, angles) -> None:
        """QAOA+ circuit: single feasible init state, then cost + mixer layers."""
        gammas, betas = base.split_angles(
            angles, self.num_gamma, self.num_beta, self.angle_strategy
        )

        for wire, bit in enumerate(self.initial_state):
            if bit == "1":
                qml.PauliX(wires=wire)

        for q in range(self.n_layers):
            base.apply_cost_unitary(self.problem_ham, gammas, q)
            for move_idx, move in enumerate(self.mixer_moves):
                beta = betas[q][move_idx]
                wires = move["wires"]
                for control_wires, control_values in move["clauses"]:
                    gate = qml.RX if move["kind"] == "flip" else qml.IsingXY
                    if control_wires:
                        qml.ctrl(
                            gate, control=control_wires, control_values=control_values
                        )(beta, wires=wires)
                    else:
                        gate(beta, wires=wires)

    # ------------------------------------------------------------------
    # Compiled functions
    # ------------------------------------------------------------------

    def _build_compiled_fns(self):
        dev = qml.device("default.qubit", wires=self.all_wires)
        problem_ham = self.problem_ham

        @qml.qnode(dev, interface="jax")
        def cost_circuit(angles):
            self.circuit(angles)
            return qml.expval(problem_ham)

        optimizer = optax.adam(self.learning_rate)

        @jax.jit
        def step(angles, opt_state):
            cost, grads = jax.value_and_grad(lambda a: jnp.real(cost_circuit(a)))(angles)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_angles = optax.apply_updates(angles, updates)
            return new_angles, new_opt_state, cost

        compiled_cost = jax.jit(lambda a: jnp.real(cost_circuit(a)))

        return compiled_cost, step

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize_angles(
        self,
        cost_fn=None,           # ignored -- kept for API compatibility
        maximize: bool = False,
        prev_layer_angles=None,
    ) -> Tuple[float, any]:
        optimizer = optax.adam(self.learning_rate)
        best_cost = float("inf")
        best_angles = None
        sign = -1.0 if maximize else 1.0

        total_params = (self.num_gamma + self.num_beta
                        if self.angle_strategy == "ma-QAOA" else 2)

        if prev_layer_angles is not None:
            new_size = self.n_layers * total_params - prev_layer_angles.size
            starting_angles = jnp.concatenate([
                prev_layer_angles.flatten(),
                jnp.zeros(new_size),
            ]).reshape(self.n_layers, total_params)
        else:
            starting_angles = None

        start_wall = time.time()
        key = jax.random.PRNGKey(int(time.time() * 1e6) % (2 ** 32))

        for restart_idx in range(self.num_restarts):
            key, subkey = jax.random.split(key)
            shape = (self.n_layers, total_params)

            if starting_angles is not None and restart_idx == 0:
                angles = starting_angles
            elif prev_layer_angles is not None:
                new_vals = jax.random.uniform(
                    subkey,
                    (self.n_layers * total_params - prev_layer_angles.size,),
                    minval=-2 * jnp.pi, maxval=2 * jnp.pi,
                )
                angles = jnp.concatenate(
                    [prev_layer_angles.flatten(), new_vals]
                ).reshape(shape)
            else:
                angles = jax.random.uniform(
                    subkey, shape, minval=-2 * jnp.pi, maxval=2 * jnp.pi
                )

            opt_state = optimizer.init(angles)

            for _ in range(self.steps):
                angles, opt_state, _ = self._compiled_step(angles, opt_state)

            final_cost = float(sign * self._compiled_cost(angles))
            if final_cost < best_cost:
                best_cost = final_cost
                best_angles = angles

        self.optimize_time = time.time() - start_wall
        self.opt_angles = best_angles
        return sign * best_cost, best_angles

    def solve(self) -> Tuple[float, dict, any]:
        """Run the full QAOA+: optimise angles, then sample."""
        opt_cost, opt_angles = self.optimize_angles()
        counts = self.do_counts_circuit()
        return opt_cost, counts, opt_angles

    # ------------------------------------------------------------------
    # Sampling (not jitted -- shots-based, called once)
    # ------------------------------------------------------------------

    def do_counts_circuit(self, angles=None, shots: int = 1000) -> dict:
        _angles = self.opt_angles if angles is None else angles

        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            self.circuit(_angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    def opt_circuit(self) -> None:
        """Apply the optimised circuit (for use as a subroutine)."""
        self.circuit(self.opt_angles)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def check_feasibility(self, bitstring: str) -> bool:
        return ch.check_feasibility(bitstring, self.parsed_constraints, self.n_x)

    def get_circuit_resources(self) -> Tuple:
        est_shots, est_error, group_shots, group_error = (
            base.estimate_hamiltonian_resources(self.problem_ham)
        )
        return None, est_shots, est_error, group_shots, group_error
