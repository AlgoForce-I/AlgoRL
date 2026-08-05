"""HyperCEZ learner: EfficientZero losses through task-conditioned deltas."""

from __future__ import annotations

import copy
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorl.agents.configs import HyperCEZConfig
from algorl.backends.jax.learners.efficientzero.learner import (
    EfficientZeroLearner,
    _dummy_training_batch,
    _loss_from_batch,
    _pad_burst_batches,
    _pad_burst_scan_inputs,
    _prepare_training_batch,
    _stack_training_batches,
    _strip_obs_running_count,
    _zeroed_step_metrics,
)
from algorl.backends.jax.learners.hypercez.train_state import (
    build_train_state,
    join_live_ez,
)
from algorl.backends.jax.nn.efficientzero.model import EfficientZero as EfficientZeroNetwork
from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.nn.efficientzero.obs_norm import (
    compute_tentative_obs_stats_jax,
    with_representation_obs_stats,
)
from algorl.backends.jax.nn.hypercez.hyper_model import (
    apply_hypernetwork,
    component_outputs_to_tree,
)
from algorl.backends.jax.nn.hypercez.materialize import materialize_ez_params
from algorl.backends.jax.nn.hypercez.regularizer import (
    RegTargets,
    calc_component_reg_loss,
    snapshot_reg_targets,
)
from algorl.backends.jax.planners.efficientzero import EfficientZeroPlanner
from algorl.backends.jax.world_models.hypercez import HyperCEZWorldModel
from algorl.buffers.efficientzero import EfficientZeroReplayBuffer
from algorl.core.component_context import ComponentContext
from algorl.core.replay_buffer import ReplayBuffer

_EMA_MOMENTUM = 0.99


def _task_alphas_from_state(
    alphas: dict[int, dict[str, jnp.ndarray]],
    task_id: jnp.ndarray | int,
    hnet_components: tuple[str, ...],
    num_tasks: int,
) -> dict[str, jnp.ndarray]:
    # Static Python task_id (jitted via partial): index one task only.
    if isinstance(task_id, (int, np.integer)):
        task_alphas = alphas[int(task_id)]
        return {component: task_alphas[component] for component in hnet_components}
    alpha_stack = jnp.stack(
        [
            jnp.stack([alphas[task][component] for component in hnet_components])
            for task in range(num_tasks)
        ]
    )
    task_alpha_vec = alpha_stack[task_id]
    return {
        component: task_alpha_vec[index]
        for index, component in enumerate(hnet_components)
    }


def _path_leaf_name(path: tuple[Any, ...]) -> Any:
    if not path:
        return None
    key = path[0]
    return key.key if hasattr(key, "key") else key


def _mask_hnet_grads_embedding_row_only(
    hnet_grad: Any,
    task_id: jnp.ndarray | int,
) -> Any:
    def mask(path: tuple[Any, ...], grad: jnp.ndarray) -> jnp.ndarray:
        if _path_leaf_name(path) == "task_embeddings" and len(path) == 1:
            zero = jnp.zeros_like(grad)
            return zero.at[task_id].set(grad[task_id])
        return jnp.zeros_like(grad)

    return jax.tree_util.tree_map_with_path(mask, hnet_grad)


def _mask_hnet_grads_theta_only(
    hnet_grad: Any,
    *,
    task_id: int,
    plastic_prev_tembs: bool,
) -> Any:
    def mask(path: tuple[Any, ...], grad: jnp.ndarray) -> jnp.ndarray:
        if _path_leaf_name(path) == "task_embeddings" and len(path) == 1:
            if not plastic_prev_tembs:
                return jnp.zeros_like(grad)
            keep = jnp.arange(grad.shape[0]) < task_id
            return jnp.where(keep[:, None], grad, jnp.zeros_like(grad))
        return grad

    return jax.tree_util.tree_map_with_path(mask, hnet_grad)


def _mask_task_phase_grads(
    grads: dict[str, Any],
    *,
    task_id: jnp.ndarray | int,
    hnet_components: tuple[str, ...],
    defer_theta: bool,
) -> dict[str, Any]:
    """Optionally restrict task-loss updates to main + current embedding."""
    if not defer_theta:
        return grads
    masked_hnets = {
        component: _mask_hnet_grads_embedding_row_only(
            grads["hnets"][component],
            task_id,
        )
        for component in hnet_components
    }
    return {
        **grads,
        "hnets": masked_hnets,
        "alphas": jax.tree.map(jnp.zeros_like, grads["alphas"]),
    }


def _theta_grads_for_lookahead(
    hnet_grads: dict[str, Any],
    *,
    hnet_components: tuple[str, ...],
    task_id: int,
    plastic_prev_tembs: bool,
) -> dict[str, Any]:
    return {
        component: _mask_hnet_grads_theta_only(
            hnet_grads[component],
            task_id=task_id,
            plastic_prev_tembs=plastic_prev_tembs,
        )
        for component in hnet_components
    }


def _clip_tree_by_global_norm(tree: Any, max_norm: float, *, safe_max: float = 100.0) -> Any:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return tree
    global_norm = optax.tree.norm(tree)
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    clipped = jax.tree.map(lambda leaf: leaf * scale, tree)
    return jax.tree.map(
        lambda leaf: jnp.clip(leaf, -safe_max, safe_max),
        clipped,
    )


def calc_delta_theta(
    hnet_params: dict[str, Any],
    theta_grads: dict[str, Any],
    *,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    lr_hyper: float,
    dt_scale: float,
    max_norm: float,
    use_sgd_change: bool,
) -> dict[str, Any]:
    """Lookahead hypernet change from task-loss theta grads (does not advance opt state)."""
    if use_sgd_change:
        updates = jax.tree.map(lambda grad: -lr_hyper * grad, theta_grads)
    else:
        updates, _ = optimizer.update(theta_grads, opt_state, hnet_params)
    dtheta = jax.tree.map(
        lambda update: dt_scale * jax.lax.stop_gradient(update),
        updates,
    )
    return _clip_tree_by_global_norm(dtheta, max_norm)


def materialize_from_train_state(
    train_state: dict[str, Any],
    task_id: jnp.ndarray | int,
    *,
    frozen_ez: Params,
    hnet_modules: dict[str, Any],
    hnet_components: tuple[str, ...],
    num_tasks: int,
    alpha_max: float,
) -> Params:
    """Build EZ params from Optax train state (differentiable w.r.t. train_state)."""
    live_ez = join_live_ez(
        shared=train_state["shared"],
        projections=train_state["projections"],
        frozen_ez=frozen_ez,
        hnet_components=hnet_components,
    )
    component_deltas: dict[str, Any] = {}
    for component_name in hnet_components:
        outputs = apply_hypernetwork(
            hnet_modules[component_name],
            train_state["hnets"][component_name],
            task_id,
        )
        component_deltas[component_name] = component_outputs_to_tree(
            outputs,
            frozen_ez[component_name],
        )
    task_alphas = _task_alphas_from_state(
        train_state["alphas"],
        task_id,
        hnet_components,
        num_tasks,
    )
    return materialize_ez_params(
        component_deltas=component_deltas,
        frozen_ez=frozen_ez,
        live_ez=live_ez,
        alphas=task_alphas,
        hnet_components=hnet_components,
        alpha_max=alpha_max,
    )


def _hypercez_task_step(
    train_state: dict[str, Any],
    opt_state: optax.OptState,
    obs_count: jnp.ndarray,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    lr_scale: jnp.ndarray,
    task_id: int,
    defer_theta: bool,
    *,
    materialize_params: bool,
    model: EfficientZeroNetwork,
    config: HyperCEZConfig,
    optimizer: optax.GradientTransformation,
    frozen_ez: Params,
    hnet_modules: dict[str, Any],
) -> tuple[
    dict[str, Any],
    optax.OptState,
    jnp.ndarray,
    Params,
    dict[str, jnp.ndarray],
    dict[int, dict[str, jnp.ndarray]],
    dict[str, Any],
]:
    hnet_components = config.hnet_components

    def objective(
        current_state: dict[str, Any],
    ) -> tuple[
        jnp.ndarray,
        tuple[Params, dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]:
        ez_params = materialize_from_train_state(
            current_state,
            task_id,
            frozen_ez=frozen_ez,
            hnet_modules=hnet_modules,
            hnet_components=hnet_components,
            num_tasks=config.num_tasks,
            alpha_max=config.alpha_max,
        )
        tentative_mean, tentative_var, tentative_count = compute_tentative_obs_stats_jax(
            ez_params,
            batch["observations"],
            obs_count,
        )
        forward_params = with_representation_obs_stats(
            ez_params,
            mean=tentative_mean,
            var=tentative_var,
        )
        loss, metrics = _loss_from_batch(
            forward_params,
            batch,
            model=model,
            config=config,
            rng=rng,
        )
        return loss, (forward_params, metrics, tentative_mean, tentative_var, tentative_count)

    (loss, (forward_params, metrics, tentative_mean, tentative_var, tentative_count)), grads = (
        jax.value_and_grad(objective, has_aux=True)(train_state)
    )
    alpha_grads = grads["alphas"]
    hnet_grads = grads["hnets"]

    def merge_metrics() -> dict[str, jnp.ndarray]:
        merged = _zeroed_step_metrics(batch)
        for key in merged:
            if key in metrics:
                merged[key] = metrics[key]
        merged["loss"] = loss
        return merged

    def apply_step(
        _: None,
    ) -> tuple[
        dict[str, Any],
        optax.OptState,
        jnp.ndarray,
        Params,
        dict[str, jnp.ndarray],
        dict[int, dict[str, jnp.ndarray]],
        dict[str, Any],
    ]:
        masked_grads = _mask_task_phase_grads(
            grads,
            task_id=task_id,
            hnet_components=hnet_components,
            defer_theta=defer_theta,
        )
        updates, new_opt_state = optimizer.update(masked_grads, opt_state, train_state)
        updates = jax.tree.map(lambda update: update * lr_scale, updates)
        new_state = optax.apply_updates(train_state, updates)
        shared = dict(new_state["shared"])
        rep_shared = dict(shared["representation_model"])
        rep_shared["running_mean"] = tentative_mean
        rep_shared["running_var"] = tentative_var
        shared["representation_model"] = rep_shared
        new_state = {**new_state, "shared": shared}
        # Burst scan discards params and rematerializes once after the chunk.
        if materialize_params:
            new_params = materialize_from_train_state(
                new_state,
                task_id,
                frozen_ez=frozen_ez,
                hnet_modules=hnet_modules,
                hnet_components=hnet_components,
                num_tasks=config.num_tasks,
                alpha_max=config.alpha_max,
            )
        else:
            new_params = forward_params
        return (
            new_state,
            new_opt_state,
            tentative_count,
            new_params,
            merge_metrics(),
            alpha_grads,
            hnet_grads,
        )

    def skip_step(
        _: None,
    ) -> tuple[
        dict[str, Any],
        optax.OptState,
        jnp.ndarray,
        Params,
        dict[str, jnp.ndarray],
        dict[int, dict[str, jnp.ndarray]],
        dict[str, Any],
    ]:
        return (
            train_state,
            opt_state,
            obs_count,
            forward_params,
            merge_metrics(),
            alpha_grads,
            hnet_grads,
        )

    return jax.lax.cond(jnp.isfinite(loss), apply_step, skip_step, None)


def _hypercez_reg_step(
    hnet_params: dict[str, Any],
    alpha_params: dict[int, dict[str, jnp.ndarray]],
    alpha_grads: dict[int, dict[str, jnp.ndarray]],
    task_theta_grads: dict[str, Any],
    reg_targets: RegTargets,
    dtheta: dict[str, Any] | None,
    task_id: int,
    beta: jnp.ndarray,
    lr_scale: jnp.ndarray,
    hnet_opt_state: optax.OptState,
    alpha_opt_state: optax.OptState,
    *,
    hnet_modules: dict[str, Any],
    hnet_components: tuple[str, ...],
    plastic_prev_tembs: bool,
    reg_optimizer: optax.GradientTransformation,
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, jnp.ndarray]],
    optax.OptState,
    optax.OptState,
    jnp.ndarray,
]:
    def reg_objective(hnets: dict[str, Any]) -> jnp.ndarray:
        return calc_component_reg_loss(
            hnets,
            hnet_modules=hnet_modules,
            hnet_components=hnet_components,
            task_id=task_id,
            reg_targets=reg_targets,
            dtheta=dtheta,
        )

    reg_loss, reg_grads = jax.value_and_grad(reg_objective)(hnet_params)
    scaled_reg_loss = beta * reg_loss
    masked_reg = {
        component: _mask_hnet_grads_theta_only(
            reg_grads[component],
            task_id=task_id,
            plastic_prev_tembs=plastic_prev_tembs,
        )
        for component in hnet_components
    }
    # Match HyperCEZDelta: theta Adam step sees task grads + scaled reg grads.
    hnet_grads = jax.tree.map(
        lambda task_grad, reg_grad: task_grad + beta * reg_grad,
        task_theta_grads,
        masked_reg,
    )

    hnet_updates, new_hnet_opt_state = reg_optimizer.update(
        hnet_grads,
        hnet_opt_state,
        hnet_params,
    )
    alpha_updates, new_alpha_opt_state = reg_optimizer.update(
        alpha_grads,
        alpha_opt_state,
        alpha_params,
    )
    scale = lambda update: update * lr_scale
    new_hnets = optax.apply_updates(hnet_params, jax.tree.map(scale, hnet_updates))
    new_alphas = optax.apply_updates(alpha_params, jax.tree.map(scale, alpha_updates))
    return new_hnets, new_alphas, new_hnet_opt_state, new_alpha_opt_state, scaled_reg_loss


def _hypercez_zeroed_step_metrics(batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    metrics = _zeroed_step_metrics(batch)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    metrics["reg_loss"] = zero
    metrics["cl_beta"] = zero
    metrics["dtheta_norm"] = zero
    return metrics


def _with_cl_metric_placeholders(metrics: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    metrics = dict(metrics)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    metrics.setdefault("reg_loss", zero)
    metrics.setdefault("cl_beta", zero)
    metrics.setdefault("dtheta_norm", zero)
    return metrics


def _update_reg_beta_ema(
    ema_task_loss: jnp.ndarray,
    ema_reg_loss: jnp.ndarray,
    task_loss: jnp.ndarray,
    reg_loss_raw: jnp.ndarray,
    config_beta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    task_abs = jnp.abs(task_loss)
    reg_abs = jnp.abs(reg_loss_raw)
    new_ema_task = jnp.where(
        ema_task_loss <= 0.0,
        task_abs,
        _EMA_MOMENTUM * ema_task_loss + (1.0 - _EMA_MOMENTUM) * task_abs,
    )
    new_ema_reg = jnp.where(
        ema_reg_loss <= 0.0,
        reg_abs,
        _EMA_MOMENTUM * ema_reg_loss + (1.0 - _EMA_MOMENTUM) * reg_abs,
    )
    beta_dynamic = config_beta * (new_ema_task / (new_ema_reg + 1e-8))
    beta_dynamic = jnp.minimum(beta_dynamic, jnp.asarray(config_beta * 1000.0, dtype=jnp.float32))
    return new_ema_task, new_ema_reg, beta_dynamic


def _hypercez_full_train_step(
    train_state: dict[str, Any],
    opt_state: optax.OptState,
    reg_hnet_opt_state: optax.OptState,
    reg_alpha_opt_state: optax.OptState,
    obs_count: jnp.ndarray,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    lr_scale: jnp.ndarray,
    ema_task_loss: jnp.ndarray,
    ema_reg_loss: jnp.ndarray,
    *,
    task_id: int,
    defer_theta: bool,
    reg_targets: RegTargets,
    materialize_params: bool,
    model: EfficientZeroNetwork,
    config: HyperCEZConfig,
    optimizer: optax.GradientTransformation,
    reg_optimizer: optax.GradientTransformation,
    frozen_ez: Params,
    hnet_modules: dict[str, Any],
) -> tuple[
    dict[str, Any],
    optax.OptState,
    optax.OptState,
    optax.OptState,
    jnp.ndarray,
    Params,
    dict[str, jnp.ndarray],
    jnp.ndarray,
    jnp.ndarray,
]:
    (
        train_state,
        opt_state,
        obs_count,
        params,
        metrics,
        alpha_grads,
        hnet_grads,
    ) = _hypercez_task_step(
        train_state,
        opt_state,
        obs_count,
        batch,
        rng,
        lr_scale,
        task_id,
        defer_theta,
        materialize_params=materialize_params,
        model=model,
        config=config,
        optimizer=optimizer,
        frozen_ez=frozen_ez,
        hnet_modules=hnet_modules,
    )

    if not defer_theta:
        return (
            train_state,
            opt_state,
            reg_hnet_opt_state,
            reg_alpha_opt_state,
            obs_count,
            params,
            _with_cl_metric_placeholders(metrics),
            ema_task_loss,
            ema_reg_loss,
        )

    task_theta_grads = _theta_grads_for_lookahead(
        hnet_grads,
        hnet_components=config.hnet_components,
        task_id=task_id,
        plastic_prev_tembs=config.plastic_prev_tembs,
    )
    if config.no_look_ahead:
        dtheta = jax.tree.map(jnp.zeros_like, task_theta_grads)
        dtheta_for_reg = None
    else:
        dtheta = calc_delta_theta(
            train_state["hnets"],
            task_theta_grads,
            optimizer=reg_optimizer,
            opt_state=reg_hnet_opt_state,
            lr_hyper=config.lr_hyper,
            dt_scale=config.dt_scale,
            max_norm=config.hnet_grad_max_norm,
            use_sgd_change=config.use_sgd_change,
        )
        dtheta_for_reg = dtheta

    reg_loss_raw = calc_component_reg_loss(
        train_state["hnets"],
        hnet_modules=hnet_modules,
        hnet_components=config.hnet_components,
        task_id=task_id,
        reg_targets=reg_targets,
        dtheta=dtheta_for_reg,
    )
    ema_task_loss, ema_reg_loss, beta_eff = _update_reg_beta_ema(
        ema_task_loss,
        ema_reg_loss,
        metrics["loss"],
        reg_loss_raw,
        config.beta,
    )
    (
        new_hnets,
        new_alphas,
        reg_hnet_opt_state,
        reg_alpha_opt_state,
        reg_loss,
    ) = _hypercez_reg_step(
        train_state["hnets"],
        train_state["alphas"],
        alpha_grads,
        task_theta_grads,
        reg_targets,
        dtheta,
        task_id,
        beta_eff,
        lr_scale,
        reg_hnet_opt_state,
        reg_alpha_opt_state,
        hnet_modules=hnet_modules,
        hnet_components=config.hnet_components,
        plastic_prev_tembs=config.plastic_prev_tembs,
        reg_optimizer=reg_optimizer,
    )
    train_state = {**train_state, "hnets": new_hnets, "alphas": new_alphas}
    if materialize_params:
        params = materialize_from_train_state(
            train_state,
            task_id,
            frozen_ez=frozen_ez,
            hnet_modules=hnet_modules,
            hnet_components=config.hnet_components,
            num_tasks=config.num_tasks,
            alpha_max=config.alpha_max,
        )
    metrics = dict(metrics)
    metrics["reg_loss"] = reg_loss
    metrics["cl_beta"] = beta_eff
    metrics["dtheta_norm"] = (
        optax.tree.norm(dtheta)
        if not config.no_look_ahead
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    return (
        train_state,
        opt_state,
        reg_hnet_opt_state,
        reg_alpha_opt_state,
        obs_count,
        params,
        metrics,
        ema_task_loss,
        ema_reg_loss,
    )


def _hypercez_burst_scan(
    train_state: dict[str, Any],
    opt_state: optax.OptState,
    reg_hnet_opt_state: optax.OptState,
    reg_alpha_opt_state: optax.OptState,
    obs_count: jnp.ndarray,
    ema_task_loss: jnp.ndarray,
    ema_reg_loss: jnp.ndarray,
    stacked_batch: dict[str, jnp.ndarray],
    rngs: jax.Array,
    lr_scales: jax.Array,
    active_mask: jax.Array,
    *,
    task_id: int,
    defer_theta: bool,
    reg_targets: RegTargets,
    model: EfficientZeroNetwork,
    config: HyperCEZConfig,
    optimizer: optax.GradientTransformation,
    reg_optimizer: optax.GradientTransformation,
    frozen_ez: Params,
    hnet_modules: dict[str, Any],
) -> tuple[
    dict[str, Any],
    optax.OptState,
    optax.OptState,
    optax.OptState,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    dict[str, jnp.ndarray],
]:
    def scan_step(
        carry: tuple[
            dict[str, Any],
            optax.OptState,
            optax.OptState,
            optax.OptState,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        inputs: tuple[dict[str, jnp.ndarray], jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[
            dict[str, Any],
            optax.OptState,
            optax.OptState,
            optax.OptState,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        dict[str, jnp.ndarray],
    ]:
        (
            current_state,
            current_opt_state,
            current_reg_hnet_opt_state,
            current_reg_alpha_opt_state,
            current_obs_count,
            current_ema_task,
            current_ema_reg,
        ) = carry
        batch, rng, lr_scale, active = inputs

        def run_step(
            _: None,
        ) -> tuple[
            tuple[
                dict[str, Any],
                optax.OptState,
                optax.OptState,
                optax.OptState,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            dict[str, jnp.ndarray],
        ]:
            (
                new_state,
                new_opt_state,
                new_reg_hnet_opt_state,
                new_reg_alpha_opt_state,
                new_obs_count,
                _params,
                metrics,
                new_ema_task,
                new_ema_reg,
            ) = _hypercez_full_train_step(
                current_state,
                current_opt_state,
                current_reg_hnet_opt_state,
                current_reg_alpha_opt_state,
                current_obs_count,
                batch,
                rng,
                lr_scale,
                current_ema_task,
                current_ema_reg,
                task_id=task_id,
                defer_theta=defer_theta,
                reg_targets=reg_targets,
                materialize_params=False,
                model=model,
                config=config,
                optimizer=optimizer,
                reg_optimizer=reg_optimizer,
                frozen_ez=frozen_ez,
                hnet_modules=hnet_modules,
            )
            return (
                (
                    new_state,
                    new_opt_state,
                    new_reg_hnet_opt_state,
                    new_reg_alpha_opt_state,
                    new_obs_count,
                    new_ema_task,
                    new_ema_reg,
                ),
                metrics,
            )

        def skip_step(
            _: None,
        ) -> tuple[
            tuple[
                dict[str, Any],
                optax.OptState,
                optax.OptState,
                optax.OptState,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            dict[str, jnp.ndarray],
        ]:
            return carry, _hypercez_zeroed_step_metrics(batch)

        return jax.lax.cond(active > 0.0, run_step, skip_step, None)

    scan_inputs = (stacked_batch, rngs, lr_scales, active_mask)
    (
        train_state,
        opt_state,
        reg_hnet_opt_state,
        reg_alpha_opt_state,
        obs_count,
        ema_task_loss,
        ema_reg_loss,
    ), metrics = jax.lax.scan(
        scan_step,
        (
            train_state,
            opt_state,
            reg_hnet_opt_state,
            reg_alpha_opt_state,
            obs_count,
            ema_task_loss,
            ema_reg_loss,
        ),
        scan_inputs,
    )
    stacked_metrics = jax.tree.map(lambda leaf: jnp.asarray(leaf), metrics)
    return (
        train_state,
        opt_state,
        reg_hnet_opt_state,
        reg_alpha_opt_state,
        obs_count,
        ema_task_loss,
        ema_reg_loss,
        stacked_metrics,
    )


class HyperCEZLearner(EfficientZeroLearner):
    """Train HyperCEZDelta generators with EfficientZero task losses.

    On task 0 (or ``beta == 0``), one phase updates shared leaves, projections,
    alphas, and the full hypernet from the task loss.

    On later tasks with ``beta > 0``, phase A updates shared/projections and the
    current embedding only; phase B applies fix-target regularization to hypernet
    theta (with optional lookahead ``Δθ``) and deferred alpha grads.
    """

    def __init__(self, context: ComponentContext) -> None:
        if not isinstance(context.config, HyperCEZConfig):
            raise TypeError(
                "HyperCEZLearner requires HyperCEZConfig, "
                f"got {type(context.config)!r}."
            )
        if not isinstance(context.world_model, HyperCEZWorldModel):
            raise TypeError(
                "HyperCEZLearner requires HyperCEZWorldModel, "
                f"got {type(context.world_model)!r}."
            )

        self.task_id = int(context.world_model.task_id)
        self._reg_targets: RegTargets | None = None
        self._ema_task_loss: float | None = None
        self._ema_reg_loss: float | None = None

        # Shared EZ wiring (params copies, planner sync, burst spec, discrete checks).
        super().__init__(context)

        self.config: HyperCEZConfig = context.config
        self.world_model: HyperCEZWorldModel = context.world_model
        self.task_id = int(self.world_model.task_id)
        self.frozen_ez: Params = copy.deepcopy(self.world_model.frozen_ez)
        self.hnet_modules = self.world_model.hnet_modules
        self.train_state = build_train_state(
            hnet_params=copy.deepcopy(self.world_model.hnet_params),
            alphas=copy.deepcopy(self.world_model.alphas),
            live_ez=copy.deepcopy(self.world_model.live_ez),
            hnet_components=self.config.hnet_components,
        )
        self._init_delayed_hnet_copies()

        self._optimizer = self._build_task_optimizer()
        self._opt_state = self._optimizer.init(self.train_state)
        self._reg_optimizer = self._build_reg_optimizer()
        self._reg_hnet_opt_state = self._reg_optimizer.init(self.train_state["hnets"])
        self._reg_alpha_opt_state = self._reg_optimizer.init(self.train_state["alphas"])
        self._recompile_train_kernels()

    def _train_kernel_kwargs(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "defer_theta": self._defer_theta(),
            "reg_targets": self._burst_reg_targets(),
            "model": self.model,
            "config": self.config,
            "optimizer": self._optimizer,
            "reg_optimizer": self._reg_optimizer,
            "frozen_ez": self.frozen_ez,
            "hnet_modules": self.hnet_modules,
        }

    def _recompile_train_kernels(self) -> None:
        """Rebuild jitted single-step and burst kernels after task / CL changes."""
        kernel_kwargs = self._train_kernel_kwargs()
        # Rematerialize once after the jitted update (planner sync), not inside
        # the AD body — same policy for single-step and burst.
        self._single_update = jax.jit(
            partial(
                _hypercez_full_train_step,
                materialize_params=False,
                **kernel_kwargs,
            )
        )
        self._burst_update = jax.jit(
            partial(_hypercez_burst_scan, **kernel_kwargs)
        )
        self._warmup_train_compile()

    def _build_task_optimizer(self) -> optax.GradientTransformation:
        hyper_opt = optax.chain(
            optax.clip_by_global_norm(self.config.hnet_grad_max_norm),
            optax.adam(self.config.lr_hyper),
        )
        main_opt = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.add_decayed_weights(self.config.weight_decay),
            optax.adam(self.config.learning_rate),
        )
        label_tree = {
            "hnets": "hyper",
            "alphas": "hyper",
            "shared": "main",
            "projections": "main",
        }
        return optax.multi_transform(
            {"hyper": hyper_opt, "main": main_opt},
            label_tree,
        )

    def _build_reg_optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.config.hnet_grad_max_norm),
            optax.adam(self.config.lr_hyper),
        )

    def _init_delayed_hnet_copies(self) -> None:
        self._self_play_hnets = copy.deepcopy(self.train_state["hnets"])
        self._reanalyze_hnets = copy.deepcopy(self.train_state["hnets"])
        self._recent_reanalyze_hnets = copy.deepcopy(self.train_state["hnets"])
        self._refresh_materialized_copies()

    def _materialize_with_hnets(self, hnets: dict[str, Any]) -> Params:
        return materialize_from_train_state(
            {**self.train_state, "hnets": hnets},
            self.task_id,
            frozen_ez=self.frozen_ez,
            hnet_modules=self.hnet_modules,
            hnet_components=self.config.hnet_components,
            num_tasks=self.config.num_tasks,
            alpha_max=self.config.alpha_max,
        )

    def _refresh_materialized_copies(self) -> None:
        self._self_play_params = _strip_obs_running_count(
            self._materialize_with_hnets(self._self_play_hnets)
        )
        self._reanalyze_params = _strip_obs_running_count(
            self._materialize_with_hnets(self._reanalyze_hnets)
        )
        self._recent_reanalyze_params = _strip_obs_running_count(
            self._materialize_with_hnets(self._recent_reanalyze_hnets)
        )
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.self_play_params = self._self_play_params

    def _sync_delayed_hnet_copies(self) -> None:
        """Align delayed hypernet snapshots (e.g. after a task boundary)."""
        self._self_play_hnets = copy.deepcopy(self.train_state["hnets"])
        self._reanalyze_hnets = copy.deepcopy(self.train_state["hnets"])
        self._recent_reanalyze_hnets = copy.deepcopy(self.train_state["hnets"])
        self._refresh_materialized_copies()

    def _burst_reg_targets(self) -> RegTargets:
        if self._reg_targets is not None:
            return self._reg_targets
        return {component: [] for component in self.config.hnet_components}

    def _defer_theta(self) -> bool:
        return (
            self.task_id > 0
            and self.config.beta > 0.0
            and self._reg_targets is not None
        )

    def _ema_scalar(self, value: float | None) -> jnp.ndarray:
        return jnp.asarray(0.0 if value is None else value, dtype=jnp.float32)

    def _sync_ema_from_scan(self, ema_task: jnp.ndarray, ema_reg: jnp.ndarray) -> None:
        self._ema_task_loss = float(np.asarray(ema_task))
        self._ema_reg_loss = float(np.asarray(ema_reg))

    def _effective_reg_beta(self, task_loss: float, reg_loss: float) -> float:
        if self._ema_task_loss is None:
            self._ema_task_loss = abs(task_loss)
        else:
            self._ema_task_loss = (
                _EMA_MOMENTUM * self._ema_task_loss
                + (1.0 - _EMA_MOMENTUM) * abs(task_loss)
            )
        if self._ema_reg_loss is None:
            self._ema_reg_loss = abs(reg_loss)
        else:
            self._ema_reg_loss = (
                _EMA_MOMENTUM * self._ema_reg_loss
                + (1.0 - _EMA_MOMENTUM) * abs(reg_loss)
            )
        beta_dynamic = self.config.beta * (
            self._ema_task_loss / (self._ema_reg_loss + 1e-8)
        )
        return min(beta_dynamic, self.config.beta * 1000.0)

    def on_task_boundary(self, new_task_id: int) -> None:
        """Snapshot previous-task hypernet outputs and switch active task."""
        new_task_id = int(new_task_id)
        if new_task_id > 0:
            self._reg_targets = snapshot_reg_targets(
                self.train_state["hnets"],
                self.hnet_modules,
                self.config.hnet_components,
                new_task_id,
            )
        else:
            self._reg_targets = None
        self.set_task_id(new_task_id)
        # ``reg_targets`` shape/content changed even when ``task_id`` is unchanged.
        self._recompile_train_kernels()
        if new_task_id > 0:
            self._sync_delayed_hnet_copies()

    def set_task_id(self, task_id: int) -> None:
        """Switch the active task embedding used for materialize / rollouts."""
        task_id = int(task_id)
        if task_id == self.task_id:
            return
        self.task_id = task_id
        self.world_model.set_task_id(self.task_id)
        self.params = self.world_model.params
        self._sync_params()
        self._recompile_train_kernels()

    def train_step(
        self,
        replay_buffer: ReplayBuffer,
        *,
        skip_reanalyze: bool = False,
    ) -> dict[str, float]:
        if not isinstance(replay_buffer, EfficientZeroReplayBuffer):
            raise TypeError(
                "HyperCEZLearner requires EfficientZeroReplayBuffer, "
                f"got {type(replay_buffer)!r}."
            )

        beta = self._priority_beta()
        batch = replay_buffer.sample(
            self.config.batch_size,
            beta=beta,
            trained_steps=self._train_steps,
        )
        self._rng_key, step_key = jax.random.split(self._rng_key)
        arrays = _prepare_training_batch(
            batch,
            planner=self.planner,
            config=self.config,
            model=self.model,
            reanalyze_params=self._reanalyze_params,
            trained_steps=self._train_steps,
            total_transitions=replay_buffer.total_transitions,
            rng_key=step_key,
            on_reanalyze_progress=getattr(self, "_on_reanalyze_progress", None),
            skip_reanalyze=skip_reanalyze,
            value_infer_fn=self._value_infer_fn,
            reanalyze_search_width=self._reanalyze_search_width,
        )
        self._maybe_refresh_model_copies()
        lr_scale = jnp.asarray(self._learning_rate_scale(), dtype=jnp.float32)
        defer_theta = self._defer_theta()
        (
            self.train_state,
            self._opt_state,
            self._reg_hnet_opt_state,
            self._reg_alpha_opt_state,
            self._obs_running_count,
            self.params,
            metrics,
            ema_task,
            ema_reg,
        ) = self._single_update(
            self.train_state,
            self._opt_state,
            self._reg_hnet_opt_state,
            self._reg_alpha_opt_state,
            jnp.asarray(self._obs_running_count, dtype=jnp.int32),
            arrays,
            step_key,
            lr_scale,
            self._ema_scalar(self._ema_task_loss),
            self._ema_scalar(self._ema_reg_loss),
        )
        if defer_theta:
            self._sync_ema_from_scan(ema_task, ema_reg)

        self._obs_running_count = int(np.asarray(self._obs_running_count))
        self.params = materialize_from_train_state(
            self.train_state,
            self.task_id,
            frozen_ez=self.frozen_ez,
            hnet_modules=self.hnet_modules,
            hnet_components=self.config.hnet_components,
            num_tasks=self.config.num_tasks,
            alpha_max=self.config.alpha_max,
        )
        self.params = _strip_obs_running_count(self.params)
        self._propagate_obs_norm_stats()
        self._sync_params()
        priorities = np.asarray(metrics["priorities"])
        if np.all(np.isfinite(priorities)):
            replay_buffer.update_priorities(
                np.asarray(arrays["indices"]),
                priorities,
            )
        self._train_steps += 1
        return {
            key: float(value)
            for key, value in metrics.items()
            if key != "priorities"
        }

    def _run_burst_scan_chunk(
        self,
        replay_buffer: EfficientZeroReplayBuffer,
        *,
        prepared_batches: list[dict[str, jnp.ndarray]],
        step_keys: jax.Array,
        start_step: int,
        chunk_steps: int,
        on_progress: Callable[[int, int], None] | None,
        completed_steps: int,
        total_steps: int,
    ) -> dict[str, float]:
        padded_batches, active_mask = _pad_burst_batches(
            prepared_batches,
            self._burst_spec,
            active_steps=chunk_steps,
        )
        stacked_batch = _stack_training_batches(padded_batches)
        step_keys, lr_scales = _pad_burst_scan_inputs(
            step_keys,
            [self._learning_rate_scale_at(start_step + offset) for offset in range(chunk_steps)],
            spec=self._burst_spec,
        )
        defer_theta = self._defer_theta()
        (
            self.train_state,
            self._opt_state,
            self._reg_hnet_opt_state,
            self._reg_alpha_opt_state,
            self._obs_running_count,
            ema_task,
            ema_reg,
            burst_metrics,
        ) = self._burst_update(
            self.train_state,
            self._opt_state,
            self._reg_hnet_opt_state,
            self._reg_alpha_opt_state,
            jnp.asarray(self._obs_running_count, dtype=jnp.int32),
            self._ema_scalar(self._ema_task_loss),
            self._ema_scalar(self._ema_reg_loss),
            stacked_batch,
            step_keys,
            lr_scales,
            active_mask,
        )
        self._obs_running_count = int(np.asarray(self._obs_running_count))
        if defer_theta:
            self._sync_ema_from_scan(ema_task, ema_reg)
        self._train_steps = start_step + chunk_steps
        self.params = materialize_from_train_state(
            self.train_state,
            self.task_id,
            frozen_ez=self.frozen_ez,
            hnet_modules=self.hnet_modules,
            hnet_components=self.config.hnet_components,
            num_tasks=self.config.num_tasks,
            alpha_max=self.config.alpha_max,
        )
        self.params = _strip_obs_running_count(self.params)
        self._propagate_obs_norm_stats()
        for offset in range(chunk_steps):
            self._train_steps = start_step + offset + 1
            self._maybe_refresh_model_copies()
        self._train_steps = start_step + chunk_steps
        self._sync_params()

        for offset in range(chunk_steps):
            priorities = np.asarray(burst_metrics["priorities"][offset])
            if np.all(np.isfinite(priorities)):
                replay_buffer.update_priorities(
                    np.asarray(padded_batches[offset]["indices"]),
                    priorities,
                )
        if on_progress is not None:
            on_progress(completed_steps + chunk_steps, total_steps)

        return {
            key: float(burst_metrics[key][chunk_steps - 1])
            for key in burst_metrics
            if key != "priorities"
        }

    def _warmup_train_compile(self) -> None:
        if not hasattr(self, "train_state"):
            return
        spec = self._burst_spec
        self._rng_key, warmup_key = jax.random.split(self._rng_key)
        dummy_batch = _dummy_training_batch(spec, warmup_key)
        lr_scale = jnp.asarray(1.0, dtype=jnp.float32)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._single_update(
            self.train_state,
            self._opt_state,
            self._reg_hnet_opt_state,
            self._reg_alpha_opt_state,
            jnp.asarray(self._obs_running_count, dtype=jnp.int32),
            dummy_batch,
            warmup_key,
            lr_scale,
            self._ema_scalar(self._ema_task_loss),
            self._ema_scalar(self._ema_reg_loss),
        )
        if spec.burst_steps <= 1:
            return
        dummy_batches = [
            _dummy_training_batch(spec, warmup_key) for _ in range(spec.burst_steps)
        ]
        stacked_batch = _stack_training_batches(dummy_batches)
        step_keys = jax.random.split(warmup_key, spec.burst_steps)
        lr_scales = jnp.ones((spec.burst_steps,), dtype=jnp.float32)
        active_mask = jnp.ones((spec.burst_steps,), dtype=jnp.float32)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._burst_update(
            self.train_state,
            self._opt_state,
            self._reg_hnet_opt_state,
            self._reg_alpha_opt_state,
            jnp.asarray(self._obs_running_count, dtype=jnp.int32),
            self._ema_scalar(self._ema_task_loss),
            self._ema_scalar(self._ema_reg_loss),
            stacked_batch,
            step_keys,
            lr_scales,
            active_mask,
        )

    def _maybe_refresh_model_copies(self) -> None:
        self_play_interval = max(1, self.config.self_play_update_interval)
        if self._train_steps > 0 and self._train_steps % self_play_interval == 0:
            self._self_play_hnets = copy.deepcopy(self.train_state["hnets"])
            self._self_play_params = _strip_obs_running_count(
                self._materialize_with_hnets(self._self_play_hnets)
            )
            if isinstance(self.planner, EfficientZeroPlanner):
                self.planner.self_play_params = self._self_play_params

        reanalyze_interval = max(1, self.config.reanalyze_update_interval)
        if self._train_steps > 0 and self._train_steps % reanalyze_interval == 0:
            self._reanalyze_hnets = copy.deepcopy(self._recent_reanalyze_hnets)
            self._recent_reanalyze_hnets = copy.deepcopy(self.train_state["hnets"])
            self._reanalyze_params = _strip_obs_running_count(
                self._materialize_with_hnets(self._reanalyze_hnets)
            )
            self._recent_reanalyze_params = _strip_obs_running_count(
                self._materialize_with_hnets(self._recent_reanalyze_hnets)
            )

        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.params = self.params

    def _sync_params(self) -> None:
        live_ez = join_live_ez(
            shared=self.train_state["shared"],
            projections=self.train_state["projections"],
            frozen_ez=self.frozen_ez,
            hnet_components=self.config.hnet_components,
        )
        self.world_model.hnet_params = self.train_state["hnets"]
        self.world_model.alphas = self.train_state["alphas"]
        self.world_model.live_ez = live_ez
        self.world_model.task_id = self.task_id
        self.world_model.params = self.params
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.params = self.params

    def sync_self_play_for_rollout(self) -> None:
        self._propagate_obs_norm_stats()
        self._self_play_params = _strip_obs_running_count(
            self._materialize_with_hnets(self._self_play_hnets)
        )
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.self_play_params = self._self_play_params
            self.planner.params = self.params


def build_hyper_cez_learner(context: ComponentContext) -> HyperCEZLearner:
    if not isinstance(context.config, HyperCEZConfig):
        if getattr(context.config, "require_implemented", True) is False:
            from algorl.backends.jax.learners import _StubLearner

            return _StubLearner("hyper_cez", context)  # type: ignore[return-value]
        raise TypeError(
            "HyperCEZLearner requires HyperCEZConfig, "
            f"got {type(context.config)!r}."
        )
    return HyperCEZLearner(context)
