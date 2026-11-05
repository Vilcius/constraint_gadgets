"""Resumable NumPy Adam optimization for gate-budget experiments."""
from __future__ import annotations

import time

import numpy as np


class TrainingInterrupted(Exception):
    pass


def _new_angles(shape, rng, previous, restart):
    if previous is None:
        return rng.uniform(-2 * np.pi, 2 * np.pi, size=shape)
    inherited = np.asarray(previous, dtype=float).ravel()
    extra_size = int(np.prod(shape)) - inherited.size
    extra = (np.zeros(extra_size) if restart == 0 else
             rng.uniform(-2 * np.pi, 2 * np.pi, size=extra_size))
    return np.concatenate([inherited, extra]).reshape(shape)


def optimize_resumable(objective, *, shape, steps, restarts, learning_rate,
                       seed, previous=None, state=None, checkpoint=None,
                       checkpoint_steps=10, should_stop=lambda: False):
    if state is None:
        rng = np.random.default_rng(seed)
        state = {
            "restart": 0, "step": 0, "rng_state": rng.bit_generator.state,
            "angles": None, "m": None, "v": None,
            "best_angles": None, "best_loss": float("inf"),
            "history": [], "restart_results": [],
            "compile_seconds": 0.0, "optimize_seconds": 0.0,
        }
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    def save():
        state["rng_state"] = rng.bit_generator.state
        if checkpoint is not None:
            checkpoint(state)

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    while state["restart"] < restarts:
        if should_stop():
            save()
            raise TrainingInterrupted("Stopped before optimizer step")
        if state["angles"] is None:
            state["angles"] = _new_angles(
                shape, rng, previous, state["restart"])
            state["m"] = np.zeros(shape, dtype=float)
            state["v"] = np.zeros(shape, dtype=float)
            state["step"] = 0
            save()
        while state["step"] < steps:
            begin = time.perf_counter()
            loss, gradient = objective(state["angles"])
            step_number = state["step"] + 1
            state["m"] = beta1 * state["m"] + (1 - beta1) * gradient
            state["v"] = beta2 * state["v"] + (1 - beta2) * gradient ** 2
            m_hat = state["m"] / (1 - beta1 ** step_number)
            v_hat = state["v"] / (1 - beta2 ** step_number)
            state["angles"] = state["angles"] - learning_rate * m_hat / (
                np.sqrt(v_hat) + epsilon)
            state["step"] = step_number
            state["optimize_seconds"] += time.perf_counter() - begin
            if step_number == 1 or step_number % checkpoint_steps == 0 or step_number == steps:
                state["history"].append({
                    "restart": int(state["restart"]),
                    "step": step_number - 1,
                    "loss": float(loss),
                })
            stop = should_stop()
            if step_number % checkpoint_steps == 0 or step_number == steps or stop:
                save()
            if stop:
                raise TrainingInterrupted("Stopped after optimizer step")
        final_loss, _ = objective(state["angles"])
        if not np.isfinite(final_loss):
            raise FloatingPointError("Non-finite final optimizer loss")
        if final_loss < state["best_loss"]:
            state["best_loss"] = float(final_loss)
            state["best_angles"] = state["angles"].copy()
        state["restart_results"].append({
            "restart": int(state["restart"]), "final_loss": float(final_loss)})
        state["restart"] += 1
        state["step"] = 0
        state["angles"] = state["m"] = state["v"] = None
        save()
    return state
