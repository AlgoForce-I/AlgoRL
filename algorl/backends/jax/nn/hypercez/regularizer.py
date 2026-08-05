"""Continual-learning regularizers for HyperCEZ hypernetwork outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from algorl.backends.jax.nn.hypercez.hyper_model import apply_hypernetwork

RegTargets = dict[str, list[tuple[jnp.ndarray, ...]]]


def _flatten_outputs(outputs: Sequence[jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate([jnp.ravel(leaf) for leaf in outputs])


def snapshot_reg_targets(
    hnet_params: dict[str, Any],
    hnet_modules: dict[str, Any],
    hnet_components: tuple[str, ...],
    task_id: int,
) -> RegTargets:
    """Snapshot hypernet outputs for all ``j < task_id`` at a task boundary.

    Each target tuple matches ``apply_hypernetwork`` outputs for that task id.
    Targets are detached from the autodiff graph.
    """
    if task_id <= 0:
        return {component: [] for component in hnet_components}

    targets: RegTargets = {component: [] for component in hnet_components}
    for component_name in hnet_components:
        module = hnet_modules[component_name]
        params = hnet_params[component_name]
        for previous_task in range(task_id):
            outputs = apply_hypernetwork(module, params, previous_task)
            targets[component_name].append(
                tuple(jax.lax.stop_gradient(output) for output in outputs)
            )
    return targets


def stack_reg_targets(
    targets: list[tuple[jnp.ndarray, ...]],
) -> tuple[jnp.ndarray, ...]:
    """Stack per-task target tuples into a pytree of ``(num_prev, ...)`` arrays."""
    if not targets:
        return ()
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *targets)


def calc_fix_target_reg(
    hnet_params: Any,
    *,
    hnet_module: Any,
    task_id: int,
    targets: list[tuple[jnp.ndarray, ...]],
    dtheta: Any | None = None,
    reg_scaling: jnp.ndarray | None = None,
    return_per_task: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """MSE between stored targets and current hypernet outputs for ``j < task_id``.

    Previous tasks are evaluated with ``jax.vmap`` (no Python loop in the
    compiled region). When ``dtheta`` is provided it is added to hypernet params
    before the forward pass (lookahead regularization).

    ``reg_scaling``, when given, must have shape ``(task_id,)`` and multiplies
    each previous-task term before averaging (HyperCEZDelta inv-EMA scaling).
    """
    if task_id <= 0:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        if return_per_task:
            return zero, jnp.zeros((0,), dtype=jnp.float32)
        return zero
    if len(targets) != task_id:
        raise ValueError(
            f"Expected {task_id} target entries, got {len(targets)}."
        )

    stacked = stack_reg_targets(targets)
    prev_ids = jnp.arange(task_id, dtype=jnp.int32)

    def one_task(tid: jnp.ndarray) -> jnp.ndarray:
        predicted = apply_hypernetwork(
            hnet_module,
            hnet_params,
            tid,
            dtheta=dtheta,
        )
        target = jax.tree.map(lambda leaf: leaf[tid], stacked)
        return jnp.sum((_flatten_outputs(predicted) - _flatten_outputs(target)) ** 2)

    per_task = jax.vmap(one_task)(prev_ids)
    if reg_scaling is not None:
        per_task = per_task * reg_scaling
    reg = jnp.mean(per_task)
    if return_per_task:
        return reg, per_task
    return reg


def calc_component_reg_loss(
    hnet_params: dict[str, Any],
    *,
    hnet_modules: dict[str, Any],
    hnet_components: tuple[str, ...],
    task_id: int,
    reg_targets: RegTargets,
    dtheta: dict[str, Any] | None = None,
    reg_scaling: jnp.ndarray | None = None,
    return_per_task: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Mean fix-target regularization over all hypernet components.

    When ``return_per_task`` is True, also returns the mean per-previous-task
    reg vector (averaged across components) for inv-EMA scaling updates.
    """
    if task_id <= 0:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        if return_per_task:
            return zero, jnp.zeros((0,), dtype=jnp.float32)
        return zero

    total = jnp.asarray(0.0, dtype=jnp.float32)
    per_task_acc = jnp.zeros((task_id,), dtype=jnp.float32)
    n_components = jnp.asarray(len(hnet_components), dtype=jnp.float32)
    for component_name in hnet_components:
        component_dtheta = None if dtheta is None else dtheta.get(component_name)
        component_reg, component_per_task = calc_fix_target_reg(
            hnet_params[component_name],
            hnet_module=hnet_modules[component_name],
            task_id=task_id,
            targets=reg_targets[component_name],
            dtheta=component_dtheta,
            reg_scaling=reg_scaling,
            return_per_task=True,
        )
        total = total + component_reg
        per_task_acc = per_task_acc + component_per_task
    mean_reg = total / n_components
    if return_per_task:
        return mean_reg, per_task_acc / n_components
    return mean_reg


def reg_scaling_from_ema(
    ema_per_task: jnp.ndarray,
    task_id: int,
    *,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
) -> jnp.ndarray:
    """Inverse-EMA per-task weights, normalized and clamped (HyperCEZDelta)."""
    if task_id <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    vals = ema_per_task[:task_id]
    # Uninitialized bins (0) → treat as 1 so scaling stays neutral until warm.
    vals = jnp.where(vals <= 0.0, jnp.ones_like(vals), vals)
    inv = 1.0 / (vals + 1e-8)
    inv = inv / (jnp.mean(inv) + 1e-8)
    return jnp.clip(inv, scale_min, scale_max)


def update_per_task_reg_ema(
    ema_per_task: jnp.ndarray,
    per_task_regs: jnp.ndarray,
    task_id: int,
    *,
    momentum: float = 0.99,
) -> jnp.ndarray:
    """EMA-update the first ``task_id`` bins of ``ema_per_task``."""
    if task_id <= 0:
        return ema_per_task
    abs_regs = jnp.abs(per_task_regs)
    prev = ema_per_task[:task_id]
    uninitialized = prev <= 0.0
    updated = jnp.where(
        uninitialized,
        abs_regs,
        momentum * prev + (1.0 - momentum) * abs_regs,
    )
    return ema_per_task.at[:task_id].set(updated)
