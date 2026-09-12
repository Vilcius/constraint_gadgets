"""Seeded, resumable Adam loop shared by the opt-in tuning workflow.

Circuit construction stays in VCG/PenaltyQAOA. Checkpoints are supplied by the
caller; no filesystem paths, study definitions, or output formats live here.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax


class TrainingInterrupted(Exception):
    """Execution stopped at an optimizer boundary with state saved."""


def make_adam_functions(objective, learning_rate):
    optimizer = optax.adam(learning_rate)

    @jax.jit
    def step(angles, opt_state, *args):
        value, gradient = jax.value_and_grad(objective)(angles, *args)
        updates, opt_state = optimizer.update(gradient, opt_state)
        return optax.apply_updates(angles, updates), opt_state, value

    return jax.jit(objective), step


def optimize_resumable(cost, step_fn, *, shape, steps, restarts,
                       learning_rate, seed, args=(), previous=None, starting=None,
                       state=None, checkpoint=None, checkpoint_steps=10,
                       zero_pad_first=True,
                       should_stop=lambda: False):
    """Follow the existing restart/angle conventions, retaining final losses.

    State includes the *next* step/restart to execute. At the first restart,
    inherited angles are zero-padded if zero_pad_first is true (PenaltyQAOA).
    VCG uses random new angles at every inherited-depth restart.
    The caller can checkpoint the returned state or stop in its callback.
    JIT warm-up uses a disposable optimizer step, never advancing saved state.
    """
    optimizer = optax.adam(learning_rate)
    if state is None:
        state = dict(restart=0, step=0, key=jax.random.PRNGKey(seed),
                     angles=None, opt_state=None, best_angles=None,
                     best_loss=float('inf'), history=[], restart_results=[],
                     compile_seconds=0.0, optimize_seconds=0.0)
    else:
        state = jax.tree.map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, state)

    def save():
        if checkpoint is not None:
            checkpoint(state)

    compiled_here = False
    while state['restart'] < restarts:
        if should_stop():
            save()
            raise TrainingInterrupted('Stopped before next optimizer step')
        if state['angles'] is None:
            state['key'], subkey = jax.random.split(state['key'])
            if state['restart'] == 0 and starting is not None:
                angles = jnp.asarray(starting).reshape(shape)
            elif previous is not None:
                inherited = jnp.asarray(previous).ravel()
                length = int(np.prod(shape)) - inherited.size
                extra = (jnp.zeros(length) if zero_pad_first and state['restart'] == 0 else
                         jax.random.uniform(subkey, (length,),
                                            minval=-2*jnp.pi, maxval=2*jnp.pi))
                angles = jnp.concatenate([inherited, extra]).reshape(shape)
            else:
                angles = jax.random.uniform(subkey, shape,
                                            minval=-2*jnp.pi, maxval=2*jnp.pi)
            state['angles'] = angles
            state['opt_state'] = optimizer.init(angles)
            state['step'] = 0
            save()
        if not compiled_here:
            begin = time.perf_counter()
            dummy = step_fn(state['angles'], state['opt_state'], *args)
            jax.block_until_ready(dummy)
            jax.block_until_ready(cost(state['angles'], *args))
            state['compile_seconds'] += time.perf_counter() - begin
            compiled_here = True
        while state['step'] < steps:
            begin = time.perf_counter()
            angles, opt_state, before_loss = step_fn(
                state['angles'], state['opt_state'], *args)
            jax.block_until_ready(angles)
            state['optimize_seconds'] += time.perf_counter() - begin
            state['angles'], state['opt_state'] = angles, opt_state
            state['step'] += 1
            # The loss is evaluated BEFORE this update. Final losses below are
            # evaluated AFTER the final update and accompany saved angles.
            if state['step'] == 1 or state['step'] % checkpoint_steps == 0 or state['step'] == steps:
                state['history'].append(dict(restart=int(state['restart']),
                                            step=int(state['step']) - 1,
                                            loss=float(before_loss)))
            stop = should_stop()
            if state['step'] % checkpoint_steps == 0 or state['step'] == steps or stop:
                save()
            if stop:
                raise TrainingInterrupted('Stopped after optimizer step')
        final_loss = float(cost(state['angles'], *args))
        if not np.isfinite(final_loss):
            raise FloatingPointError('Non-finite final optimizer loss')
        if final_loss < state['best_loss']:
            state['best_loss'] = final_loss
            state['best_angles'] = state['angles']
        state['restart_results'].append(dict(restart=int(state['restart']),
                                             final_loss=final_loss))
        state['restart'] += 1
        state['step'] = 0
        state['angles'], state['opt_state'] = None, None
        save()
    return state
