"""
vcg.py -- Variational Constraint Gadget (VCG).

Uses QAOA to prepare a superposition of states satisfying a constraint.
The Hamiltonian is defined directly on the n_x decision-variable qubits,
assigning eigenvalue -1 to feasible assignments and +1 to infeasible ones.

Training procedure:
1. Single QAOA p=1 run to get warm-start angles.
2. ma-QAOA layer sweep, warm-starting each new layer from the previous
   depth's optimal angles, until AR >= ar_threshold or max_layers.

The only public training entry point is train().  Raw angle optimisation
is intentionally not exposed.
"""

import itertools as it
import math
import time
from math import comb

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from pennylane import numpy as np

from . import qaoa_base as base
from . import constraint_handler as ch
from .dicke_state_prep import prepare_dicke_multiweight_state


def _check_constraint_op(lhs_val: float, op: ch.ConstraintOp, rhs: float) -> bool:
    """Return True if ``lhs_val op rhs`` holds."""
    if op == ch.ConstraintOp.EQ:
        return abs(lhs_val - rhs) < 1e-6
    if op == ch.ConstraintOp.LEQ:
        return lhs_val <= rhs + 1e-6
    if op == ch.ConstraintOp.LT:
        return lhs_val < rhs - 1e-6
    if op == ch.ConstraintOp.GEQ:
        return lhs_val >= rhs - 1e-6
    if op == ch.ConstraintOp.GT:
        return lhs_val > rhs + 1e-6
    return False


class VCG:
    """
    Variational Constraint Gadget (VCG).

    Parameters
    ----------
    constraints : list[str]
        Constraint strings (zero-indexed variable names).
    ar_threshold : float
        Stop training when AR >= this value.
    max_layers : int
        Maximum ma-QAOA layers.
    qaoa_restarts, qaoa_steps : int
        Budget for the initial QAOA p=1 run.
    ma_restarts, ma_steps : int
        Budget per ma-QAOA layer.
    lr : float
        Adam learning rate.
    samples : int
        Measurement shots for the counts circuit.
    decompose : bool
        If True, use Pauli decomposition via WHT (recommended).

    Attributes
    ----------
    var_wires : list[int]
        Decision-variable qubit indices (sorted).
    all_wires : list[int]
        Same as var_wires -- no ancilla qubits.
    n_x : int
        Number of decision variables.
    constraint_Ham : qml.Hamiltonian
        Hamiltonian with -1 for feasible, +1 for infeasible states.
    opt_angles : np.ndarray
        Set by train().
    n_layers : int
        Number of ma-QAOA layers used, set by train().
    ar : float
        Best AR achieved, set by train().
    """

    def __init__(
        self,
        constraints: list,
        ar_threshold: float = 0.999,
        entropy_threshold: float = 0.9,
        max_layers: int = 8,
        qaoa_restarts: int = 5,
        qaoa_steps: int = 150,
        ma_restarts: int = 20,
        ma_steps: int = 200,
        lr: float = 0.05,
        samples: int = 10_000,
        decompose: bool = True,
    ) -> None:
        self.constraints = constraints
        self.ar_threshold = ar_threshold
        self.entropy_threshold = entropy_threshold
        self.max_layers = max_layers
        self.qaoa_restarts = qaoa_restarts
        self.qaoa_steps = qaoa_steps
        self.ma_restarts = ma_restarts
        self.ma_steps = ma_steps
        self.lr = lr
        self.samples = samples
        self.decompose = decompose

        self.parsed_constraints = ch.parse_constraints(self.constraints)
        all_vars = sorted(set().union(*(pc.variables for pc in self.parsed_constraints)))
        self.var_wires = all_vars
        self.all_wires = all_vars
        self.n_x = len(all_vars)
        self._wire_to_idx = {w: i for i, w in enumerate(self.var_wires)}

        self.outcomes = self._make_outcomes()
        n_feasible = self.outcomes.count(-1.0)
        self.n_feasible = n_feasible
        if n_feasible == 0:
            raise ValueError(
                f"VCG: constraint(s) {self.constraints} have no feasible "
                f"binary assignments. Check your constraint."
            )
        # Single feasible state: record it for X-gate preparation, skip QAOA.
        self._single_feasible_bitstring = None
        if n_feasible == 1:
            idx = self.outcomes.index(-1.0)
            self._single_feasible_bitstring = format(idx, f'0{self.n_x}b')

        # Dicke-superposition: feasible set = union of complete Hamming-weight
        # classes → exact state preparation, no QAOA needed.
        self._dicke_superposition_weights = None
        if n_feasible > 1:
            self._dicke_superposition_weights = self._check_dicke_superposition()

        self.constraint_Ham = self._build_hamiltonian()

        # Set after train()
        self.opt_angles = None
        self.n_layers = None
        self.ar = None
        self.entropy = None
        self.train_time = None   # wall-clock seconds for the full train() call

    # ------------------------------------------------------------------
    # Training (the only public path to set opt_angles)
    # ------------------------------------------------------------------

    def train(self, verbose: bool = True) -> float:
        """
        Train the gadget following the prescribed two-stage procedure.
        Sets self.train_time to the total wall-clock seconds elapsed.

        1. QAOA p=1: optimize 2 parameters (one gamma, one beta) to get
           warm-start angles.
        2. ma-QAOA layer sweep: sweep p=1,2,... warm-starting each depth
           from the previous layer's optimal angles.  Stop when
           AR >= ar_threshold or max_layers is reached.

        Sets self.opt_angles, self.n_layers, self.ar.

        Returns
        -------
        float : best AR achieved.
        """
        _train_start = time.time()

        # ── Special case: single feasible state → X-gate preparation ──
        if self._single_feasible_bitstring is not None:
            if verbose:
                print(f'  Single feasible state |{self._single_feasible_bitstring}> '
                      f'— using X-gate preparation (no QAOA needed).')
            self.opt_angles = None   # not used; opt_circuit handles this case
            self.n_layers = 0
            self.ar = 1.0
            self.entropy = 1.0       # trivially uniform over 1 feasible state
            self.num_gamma = 0
            self.num_beta = 0
            self.train_time = time.time() - _train_start
            return 1.0

        # ── Special case: Dicke superposition → exact prep, no QAOA ──
        if self._dicke_superposition_weights is not None:
            if verbose:
                print(f'  Dicke-superposition detected: weights={self._dicke_superposition_weights} '
                      f'— using exact state preparation (no QAOA needed).')
            self.opt_angles = None
            self.n_layers = 0
            self.ar = 1.0
            self.entropy = 1.0
            self.num_gamma = 0
            self.num_beta = 0
            self.train_time = time.time() - _train_start
            return 1.0

        # ── Step 1: QAOA p=1 ─────────────────────────────────────────
        if verbose:
            print(f'  QAOA p=1  (restarts={self.qaoa_restarts}, steps={self.qaoa_steps})')

        qaoa_cost, qaoa_angles = self._optimize(
            angle_strategy='QAOA',
            n_layers=1,
            steps=self.qaoa_steps,
            num_restarts=self.qaoa_restarts,
        )
        qaoa_ar = (float(qaoa_cost) - 1.0) / -2.0
        if verbose:
            print(f'    AR={qaoa_ar:.4f}')

        # ── Step 2: ma-QAOA layer sweep ───────────────────────────────
        if verbose:
            print(f'  ma-QAOA sweep  (restarts={self.ma_restarts}, steps={self.ma_steps})')

        best_ar = 0.0
        best_entropy = -1.0
        best_angles = None
        best_n_layers = 1
        prev_angles = None

        for p in range(1, self.max_layers + 1):
            num_gamma = len(self.constraint_Ham.ops) if self.decompose else 1
            num_beta = self.n_x

            if prev_angles is None:
                # p=1: seed first restart from broadcast QAOA angles
                starting = base.convert_qaoa_to_ma_angles(
                    qaoa_angles, num_gamma, num_beta, n_layers=1
                )
            else:
                starting = None  # handled by prev_layer_angles warm-start

            opt_cost, opt_angles = self._optimize(
                angle_strategy='ma-QAOA',
                n_layers=p,
                steps=self.ma_steps,
                num_restarts=self.ma_restarts,
                prev_layer_angles=prev_angles,
                starting_angles=starting,
            )
            prev_angles = opt_angles
            ar = (float(opt_cost) - 1.0) / -2.0

            # Once AR threshold is met, rank by entropy (spread over feasible states).
            # Before that, still track best AR as fallback.
            if ar >= self.ar_threshold:
                ent = self._compute_entropy_norm(opt_angles, p)
            else:
                ent = -1.0

            if ar >= self.ar_threshold and ent > best_entropy:
                best_entropy = ent
                best_angles = opt_angles
                best_n_layers = p
                best_ar = ar
            elif ar > best_ar and best_entropy < 0:
                # Fallback: AR not yet reached, track best AR
                best_ar = ar
                best_angles = opt_angles
                best_n_layers = p

            if verbose:
                ar_mark = '✓' if ar >= self.ar_threshold else ' '
                ent_str = f'  ent={ent:.4f}' if ent >= 0 else ''
                print(f'    {ar_mark} p={p}: AR={ar:.4f}{ent_str}')

            if ar >= self.ar_threshold and ent >= self.entropy_threshold:
                break

        if best_ar < self.ar_threshold and verbose:
            print(f'  [warn] Did not reach AR>={self.ar_threshold} within '
                  f'{self.max_layers} layers.  Best AR={best_ar:.4f}')
        elif best_entropy >= 0 and best_entropy < self.entropy_threshold and verbose:
            print(f'  [warn] AR threshold met but entropy below target '
                  f'(entropy={best_entropy:.4f} < {self.entropy_threshold}).  '
                  f'Best entropy used.')

        self.opt_angles = best_angles
        self.n_layers = best_n_layers
        self.ar = best_ar
        self.entropy = best_entropy if best_entropy >= 0 else None
        self.train_time = time.time() - _train_start
        # Update parameter counts to match the trained depth
        self.num_gamma = len(self.constraint_Ham.ops) if self.decompose else 1
        self.num_beta = self.n_x
        return best_ar

    # ------------------------------------------------------------------
    # Circuit interface (requires train() to have been called)
    # ------------------------------------------------------------------

    def opt_circuit(self) -> None:
        """Apply the trained circuit (HybridQAOA / Grover mixer interface)."""
        if self._single_feasible_bitstring is not None:
            for wire, bit in zip(self.var_wires, self._single_feasible_bitstring):
                if bit == '1':
                    qml.PauliX(wires=wire)
            return
        if self._dicke_superposition_weights is not None:
            prepare_dicke_multiweight_state(self.var_wires, self._dicke_superposition_weights)
            return
        if self.opt_angles is None:
            raise RuntimeError("Call train() before using opt_circuit().")
        self._circuit(self.opt_angles, 'ma-QAOA', self.n_layers)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def do_counts_circuit(self, shots: int = None) -> dict:
        """Sample measurement outcomes from the trained circuit."""
        if self.opt_angles is None and self._single_feasible_bitstring is None:
            raise RuntimeError("Call train() before sampling.")
        shots = shots or self.samples
        @qml.qnode(qml.device("default.qubit", wires=self.var_wires, shots=shots))
        def circuit():
            self.opt_circuit()
            return qml.counts(all_outcomes=True)
        start = time.time()
        counts = circuit()
        self.count_time = time.time() - start
        return counts

    def p_feasible(self, shots: int = None) -> float:
        """
        Fraction of shots that satisfy all constraints.
        """
        counts = self.do_counts_circuit(shots=shots)
        total = sum(counts.values())
        if total == 0:
            return float('nan')
        feasible = sum(v for bs, v in counts.items() if self._is_feasible(bs))
        return feasible / total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _circuit(self, angles: np.ndarray, angle_strategy: str, n_layers: int) -> None:
        """Apply QAOA circuit with given angles (internal use only)."""
        num_gamma = len(self.constraint_Ham.ops) if self.decompose else 1
        num_beta = self.n_x
        gammas, betas = base.split_angles(angles, num_gamma, num_beta, angle_strategy)

        for wire in self.var_wires:
            qml.Hadamard(wires=wire)

        for q in range(n_layers):
            if self.decompose:
                base.apply_cost_unitary(self.constraint_Ham, gammas, q)
            else:
                diag = np.exp(-1j * gammas[q][0] * np.array(self.outcomes, dtype=float))
                qml.DiagonalQubitUnitary(diag, wires=self.var_wires)
            base.apply_x_mixer(betas, q, self.var_wires)

    def _optimize(
        self,
        angle_strategy: str,
        n_layers: int,
        steps: int,
        num_restarts: int,
        prev_layer_angles=None,
        starting_angles=None,
    ) -> tuple:
        """Run JAX-jitted Adam optimisation for a given strategy/depth."""
        num_gamma = len(self.constraint_Ham.ops) if self.decompose else 1
        num_beta = self.n_x
        total_params = 2 if angle_strategy == "QAOA" else (num_gamma + num_beta)
        shape = (n_layers, total_params)

        dev = qml.device("default.qubit", wires=self.var_wires)
        constraint_ham = self.constraint_Ham

        @qml.qnode(dev, interface="jax")
        def cost_circuit(angles):
            self._circuit(angles, angle_strategy, n_layers)
            return qml.expval(constraint_ham)

        optimizer = optax.adam(self.lr)

        @jax.jit
        def step_fn(angles, opt_state):
            cost, grads = jax.value_and_grad(lambda a: jnp.real(cost_circuit(a)))(angles)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_angles = optax.apply_updates(angles, updates)
            return new_angles, new_opt_state, cost

        best_cost = float("inf")
        best_angles = None
        key = jax.random.PRNGKey(int(time.time() * 1e6) % (2 ** 32))

        for restart_idx in range(num_restarts):
            key, subkey = jax.random.split(key)

            if starting_angles is not None and restart_idx == 0:
                angles = jnp.array(starting_angles).reshape(shape)
            elif prev_layer_angles is not None:
                prev_flat = jnp.array(prev_layer_angles).flatten()
                new_vals = jax.random.uniform(
                    subkey,
                    (n_layers * total_params - prev_flat.size,),
                    minval=-2 * jnp.pi, maxval=2 * jnp.pi,
                )
                angles = jnp.concatenate([prev_flat, new_vals]).reshape(shape)
            else:
                angles = jax.random.uniform(
                    subkey, shape, minval=-2 * jnp.pi, maxval=2 * jnp.pi
                )

            opt_state = optimizer.init(angles)
            for _ in range(steps):
                angles, opt_state, cost = step_fn(angles, opt_state)

            cost_val = float(cost)
            if cost_val < best_cost:
                best_cost = cost_val
                best_angles = angles

        return best_cost, best_angles

    def _compute_entropy_norm(self, angles: np.ndarray, n_layers: int) -> float:
        """
        Normalised Shannon entropy of the distribution over feasible states.

        H_norm = H(P_{feasible}) / log(|F|)  ∈ [0, 1]

        Computed from exact statevector probabilities (qml.probs), not samples,
        so the result is noiseless and the 0.9999 threshold is meaningful even
        for small |F|.

        Returns 1.0 if |F| ≤ 1 (trivially uniform).
        """
        if self.n_feasible <= 1:
            return 1.0
        @qml.qnode(qml.device("default.qubit", wires=self.var_wires))
        def circuit():
            self._circuit(angles, 'ma-QAOA', n_layers)
            return qml.probs(wires=self.var_wires)
        probs = circuit()
        feas_total = sum(float(probs[i]) for i in range(len(probs))
                         if self.outcomes[i] == -1.0)
        if feas_total < 1e-12:
            return 0.0
        H = 0.0
        for i, p_raw in enumerate(probs):
            if self.outcomes[i] == -1.0:
                p = float(p_raw) / feas_total
                if p > 1e-12:
                    H -= p * math.log(p)
        return H / math.log(self.n_feasible)

    def _make_outcomes(self) -> list:
        """Assign -1 (feasible) / +1 (infeasible) to all 2^n_x assignments."""
        outcomes = []
        for assignment in it.product([0, 1], repeat=self.n_x):
            satisfied = True
            for pc in self.parsed_constraints:
                lhs = pc.constant
                for var_idx, coeff in pc.linear.items():
                    lhs += coeff * assignment[self._wire_to_idx[var_idx]]
                for (i, j), coeff in pc.quadratic.items():
                    lhs += (coeff * assignment[self._wire_to_idx[i]]
                            * assignment[self._wire_to_idx[j]])
                if not _check_constraint_op(lhs, pc.op, pc.rhs):
                    satisfied = False
                    break
            outcomes.append(-1.0 if satisfied else 1.0)
        return outcomes

    def _build_hamiltonian(self) -> qml.Hamiltonian:
        """Build constraint Hamiltonian via WHT Pauli decomposition."""
        start = time.time()
        n = self.n_x
        N = 1 << n
        coeffs = np.array(self.outcomes, dtype=float)

        step = 1
        while step < N:
            view = coeffs.reshape(-1, 2 * step)
            u, v = view[:, :step].copy(), view[:, step:].copy()
            view[:, :step] = u + v
            view[:, step:] = u - v
            step <<= 1
        coeffs /= N

        pauli_coeffs, pauli_ops = [], []
        # Include S=0 (identity) so eigenvalues remain in {-1, +1}
        if abs(coeffs[0]) >= 1e-10:
            pauli_coeffs.append(float(coeffs[0]))
            pauli_ops.append(qml.Identity(self.var_wires[0]))
        for S in range(1, N):
            if abs(coeffs[S]) < 1e-10:
                continue
            obs = qml.prod(*(
                qml.PauliZ(self.var_wires[n - 1 - k])
                for k in range(n) if S & (1 << k)
            ))
            pauli_coeffs.append(float(coeffs[S]))
            pauli_ops.append(obs)

        self.hamiltonian_time = time.time() - start
        return qml.Hamiltonian(np.array(pauli_coeffs), pauli_ops)

    def _check_dicke_superposition(self):
        """
        Return a sorted list of Hamming weights if the feasible set is exactly
        the union of *complete* Dicke classes at those weights — i.e. every
        bitstring of weight w is feasible (for each w in the set) and no other
        bitstrings are feasible.  Return None otherwise.

        Examples that trigger this path:
          - ``x_0 + x_1 >= 1``  (n=2): W = {1, 2}
          - ``x_0 * x_1 == 0``  (n=2): W = {0, 1}
          - ``x_0 + x_1 <= 1``  (n=2): W = {0, 1}  (already CardinalityLeq,
            but also detected here so VCG skips training)
        """
        n = self.n_x
        feas_count = {}
        for i, outcome in enumerate(self.outcomes):
            if outcome == -1.0:
                hw = bin(i).count('1')
                feas_count[hw] = feas_count.get(hw, 0) + 1
        for w, cnt in feas_count.items():
            if cnt != comb(n, w):
                return None  # partial weight class — not a Dicke superposition
        return sorted(feas_count.keys())

    def _is_feasible(self, bitstring: str) -> bool:
        assignment = [int(b) for b in bitstring]
        for pc in self.parsed_constraints:
            lhs = pc.constant
            for var_idx, coeff in pc.linear.items():
                lhs += coeff * assignment[self._wire_to_idx[var_idx]]
            for (i, j), coeff in pc.quadratic.items():
                lhs += (coeff * assignment[self._wire_to_idx[i]]
                        * assignment[self._wire_to_idx[j]])
            if not _check_constraint_op(lhs, pc.op, pc.rhs):
                return False
        return True
