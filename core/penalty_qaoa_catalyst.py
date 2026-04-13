"""
penalty_qaoa_catalyst.py -- JAX-jitted PenaltyQAOA.

Drop-in replacement for penalty_qaoa.py that JIT-compiles the cost circuit
and optimisation step via JAX (jax.jit + jax.value_and_grad).  Key
differences from the original:

  - Device: ``default.qubit`` — JAX differentiates through array ops.
  - Gradient: ``jax.value_and_grad`` (adjoint) — same method as
    HybridQAOACatalyst for direct comparability.
  - Optimizer: ``optax.adam`` instead of ``qml.AdamOptimizer``.
  - Inner step loop: fixed-length (no early-stopping ``break``).
  - Layer loop: Python ``for q in range(n_layers)`` — unrolled at JAX trace
    time.
  - numpy: ``jax.numpy`` replaces ``pennylane.numpy`` inside jitted code;
    ``requires_grad`` annotations are dropped (JAX handles AD natively).
  - Random: ``jax.random`` with explicit PRNG keys for angle initialisation.

Unchanged from the original:
  - Hamiltonian construction (runs at __init__ time, outside jit).
  - do_counts_circuit (shots-based; uses lightning.qubit without jit).
  - check_feasibility, get_circuit_resources (pure Python).
  - Public interface: solve(), optimize_angles(), do_counts_circuit() have the
    same signatures as PenaltyQAOA.

Requirements:
  pip install jax optax pennylane
"""

from __future__ import annotations

import time
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import optax
import pennylane as qml

from . import qaoa_base as base
from . import constraint_handler as ch


class PenaltyQAOACatalyst:
    """
    Catalyst-compiled penalty-based QAOA.

    Parameters match PenaltyQAOA exactly so the two are interchangeable in
    run scripts.  See penalty_qaoa.py for full parameter documentation.
    """

    def __init__(
        self,
        qubo,
        constraints: list[str],
        penalty: float = 10.0,
        angle_strategy: str = "ma-QAOA",
        n_layers: int = 1,
        samples: int = 1000,
        learning_rate: float = 0.01,
        steps: int = 50,
        num_restarts: int = 10,
    ) -> None:

        self.qubo = qubo
        self.constraints = constraints
        self.penalty_param = penalty
        self.angle_strategy = base.validate_angle_strategy(angle_strategy)
        self.n_layers = n_layers
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.num_restarts = num_restarts

        self.parsed_constraints = ch.parse_constraints(constraints)
        self.n_x = qubo.shape[0]
        self.x_wires = list(range(self.n_x))

        self.slack_info, self.n_slack = ch.determine_slack_variables(
            self.parsed_constraints, self.n_x
        )
        self.slack_wires = list(range(self.n_x, self.n_x + self.n_slack))
        self.all_wires = self.x_wires + self.slack_wires
        self.n_total = len(self.all_wires)

        # Hamiltonians built once at init, outside any jit region
        self.qubo_Ham = base.build_qubo_hamiltonian(self.qubo, self.x_wires)
        self.penalty_Ham = self._build_penalty_hamiltonian()
        self.full_Ham = self.qubo_Ham + self.penalty_Ham

        self.num_gamma = base.count_gamma_terms(self.full_Ham)
        self.num_beta = len(self.all_wires)

        # Compile cost circuit + single Adam step with Catalyst
        self._compiled_cost, self._compiled_step = self._build_compiled_fns()

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    def _build_penalty_hamiltonian(self) -> qml.Hamiltonian:
        start = time.time()
        ham = base.build_penalty_hamiltonian(
            pen_constraints=self.parsed_constraints,
            slack_infos=self.slack_info,
            delta=self.penalty_param,
            fallback_wire=self.x_wires[0],
        )
        self.hamiltonian_time = time.time() - start
        return ham

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------

    def qaoa_circuit(self, angles) -> None:
        """Standard QAOA circuit with X-mixer."""
        gammas, betas = base.split_angles(
            angles, self.num_gamma, self.num_beta, self.angle_strategy
        )

        for wire in self.all_wires:
            qml.Hadamard(wires=wire)

        for q in range(self.n_layers):
            base.apply_cost_unitary(self.full_Ham, gammas, q)
            base.apply_x_mixer(betas, q, self.all_wires)

    # ------------------------------------------------------------------
    # Compiled functions
    # ------------------------------------------------------------------

    def _build_compiled_fns(self):
        """
        Build qjit-compiled cost and step functions.

        cost(angles)             -> scalar <H>
        step(angles, opt_state)  -> (new_angles, new_opt_state, cost)

        The step function uses jax.value_and_grad to get cost + gradient in a
        single forward+backward pass, then applies the optax Adam update.
        Both functions are compiled once here; subsequent calls hit the cache.
        """
        dev = qml.device("default.qubit", wires=self.all_wires)
        full_Ham = self.full_Ham

        @qml.qnode(dev, interface="jax")
        def cost_circuit(angles):
            self.qaoa_circuit(angles)
            return qml.expval(full_Ham)

        optimizer = optax.adam(self.learning_rate)

        @jax.jit
        def step(angles, opt_state):
            cost, grads = jax.value_and_grad(cost_circuit)(angles)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_angles = optax.apply_updates(angles, updates)
            return new_angles, new_opt_state, cost

        compiled_cost = jax.jit(cost_circuit)

        return compiled_cost, step

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize_angles(
        self,
        cost_fn=None,           # ignored — kept for API compatibility
        maximize: bool = False,
        prev_layer_angles=None,
    ) -> Tuple[float, any]:
        """
        Optimise angles using compiled Adam + optax, with random restarts.

        Early stopping (conv_tol break) is removed — Catalyst cannot trace a
        conditional break inside a jitted loop.  Each restart runs the full
        ``steps`` budget.  The outer restart loop is plain Python (each
        restart is an independent compiled call, so no tracing issue).
        """
        optimizer = optax.adam(self.learning_rate)
        best_cost = float("inf")
        best_angles = None
        sign = -1.0 if maximize else 1.0

        total_params = (self.num_gamma + self.num_beta
                        if self.angle_strategy == "ma-QAOA" else 2)

        # Warm-start: inherit prev layer angles, zero-pad new layer
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
                # Append random new-layer angles to inherited prev angles
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

            # Fixed-length optimisation loop (no break — Catalyst limitation)
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
        """Run the full penalty QAOA: optimise angles, then sample."""
        opt_cost, opt_angles = self.optimize_angles()
        counts = self.do_counts_circuit()
        return opt_cost, counts, opt_angles

    # ------------------------------------------------------------------
    # Sampling (not jitted — shots-based, called once)
    # ------------------------------------------------------------------

    def do_counts_circuit(self, angles=None, shots: int = 1000) -> dict:
        """
        Sample bitstrings from the optimised circuit.

        Not compiled — shots-based measurement is called once and does not
        benefit from jit.  Uses lightning.qubit directly (no Catalyst).
        """
        _angles = self.opt_angles if angles is None else angles

        @qml.qnode(qml.device("lightning.qubit", wires=self.all_wires, shots=shots))
        def circuit():
            self.qaoa_circuit(_angles)
            return qml.counts(all_outcomes=True)

        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    # ------------------------------------------------------------------
    # Helpers (unchanged from original)
    # ------------------------------------------------------------------

    def get_circuit_resources(self, opt: bool = False) -> Tuple:
        est_shots, est_error, group_shots, group_error = (
            base.estimate_hamiltonian_resources(self.full_Ham)
        )
        return None, est_shots, est_error, group_shots, group_error

    def check_feasibility(self, bitstring: str) -> bool:
        return ch.check_feasibility(bitstring, self.parsed_constraints, self.n_x)
