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


def calc_fix_target_reg(
    hnet_params: Any,
    *,
    hnet_module: Any,
    task_id: int,
    targets: list[tuple[jnp.ndarray, ...]],
    dtheta: Any | None = None,
) -> jnp.ndarray:
    """MSE between stored targets and current hypernet outputs for ``j < task_id``.

    When ``dtheta`` is provided, it is added to hypernet params before the
    forward pass (lookahead regularization). With ``no_look_ahead=True`` the
    learner passes ``dtheta=None``.
    """
    if task_id <= 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    if len(targets) != task_id:
        raise ValueError(
            f"Expected {task_id} target entries, got {len(targets)}."
        )

    reg = jnp.asarray(0.0, dtype=jnp.float32)
    for previous_task in range(task_id):
        predicted = apply_hypernetwork(
            hnet_module,
            hnet_params,
            previous_task,
            dtheta=dtheta,
        )
        target = targets[previous_task]
        if len(predicted) != len(target):
            raise ValueError(
                f"Target/prediction length mismatch for task {previous_task}: "
                f"{len(target)} vs {len(predicted)}."
            )
        reg = reg + jnp.sum(
            ( _flatten_outputs(predicted) - _flatten_outputs(target) ) ** 2
        )
    return reg / jnp.asarray(task_id, dtype=jnp.float32)


def calc_component_reg_loss(
    hnet_params: dict[str, Any],
    *,
    hnet_modules: dict[str, Any],
    hnet_components: tuple[str, ...],
    task_id: int,
    reg_targets: RegTargets,
    dtheta: dict[str, Any] | None = None,
) -> jnp.ndarray:
    """Sum fix-target regularization over all hypernet components."""
    if task_id <= 0:
        return jnp.asarray(0.0, dtype=jnp.float32)

    total = jnp.asarray(0.0, dtype=jnp.float32)
    for component_name in hnet_components:
        component_dtheta = None if dtheta is None else dtheta.get(component_name)
        total = total + calc_fix_target_reg(
            hnet_params[component_name],
            hnet_module=hnet_modules[component_name],
            task_id=task_id,
            targets=reg_targets[component_name],
            dtheta=component_dtheta,
        )
    return total / jnp.asarray(len(hnet_components), dtype=jnp.float32)
