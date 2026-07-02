"""EfficientZero learner: unrolled value/reward/policy/consistency losses."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn.efficient_zero.losses import (
    apply_half_gradient,
    continuous_policy_loss,
    cosine_consistency_loss,
    reward_loss,
    value_loss,
)
from algorl.backends.jax.nn.efficient_zero.model import EfficientZero as EfficientZeroNetwork
from algorl.backends.jax.nn.efficient_zero.model import Params
from algorl.backends.jax.planners.mcts.efficientzero import EfficientZeroPlanner
from algorl.backends.jax.world_models.efficient_zero import EfficientZeroWorldModel
from algorl.core.component_context import ComponentContext
from algorl.core.learner import Learner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch


def _batch_to_arrays(batch: Batch) -> dict[str, jnp.ndarray]:
    required = (
        "observations",
        "actions",
        "rewards",
        "policy_targets",
        "search_values",
        "dones",
    )
    data = batch.data
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"EfficientZero learner expected batch keys {missing}.")

    arrays: dict[str, jnp.ndarray] = {}
    for key in required:
        arrays[key] = jnp.asarray(data[key], dtype=jnp.float32 if key != "dones" else jnp.bool_)
    if arrays["dones"].dtype != jnp.bool_:
        arrays["dones"] = arrays["dones"].astype(jnp.bool_)
    return arrays


def _loss_from_batch(
    params: Params,
    batch: dict[str, jnp.ndarray],
    *,
    model: EfficientZeroNetwork,
    config: EfficientZeroConfig,
    rng: jax.Array,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    search_values = batch["search_values"]
    dones = batch["dones"]

    unroll_steps = config.unroll_steps
    batch_size = observations.shape[0]

    rng, init_policy_rng, unroll_rng = jax.random.split(rng, 3)
    init_obs = observations[:, 0]

    state = model.do_representation(params, init_obs)
    values, policy = model.do_value_policy_prediction(params, state)

    value_loss_total = value_loss(values, search_values[:, 0], config)
    policy_loss_total, entropy_total = continuous_policy_loss(
        policy,
        actions[:, 0],
        entropy_rng=init_policy_rng,
    )
    reward_loss_total = jnp.zeros((batch_size,), dtype=jnp.float32)
    consistency_loss_total = jnp.zeros((batch_size,), dtype=jnp.float32)

    reward_hidden = None

    def step_fn(
        carry: tuple[jnp.ndarray, Any],
        step_inputs: tuple[jnp.ndarray, ...],
    ) -> tuple[tuple[jnp.ndarray, Any], dict[str, jnp.ndarray]]:
        state, reward_hidden = carry
        action, target_reward, target_search_value, next_obs, step_mask, policy_mask, best_action, step_rng = (
            step_inputs
        )

        next_state = model.do_dynamics(params, state, action)
        value_prefix, next_reward_hidden = model.do_reward_prediction(
            params,
            next_state,
            reward_hidden,
        )
        values, policy = model.do_value_policy_prediction(params, next_state)

        gt_state = model.do_representation(params, next_obs)
        dynamic_proj = model.do_projection(params, next_state, with_grad=True)
        gt_proj = model.do_projection(params, gt_state, with_grad=False)

        step_consistency = cosine_consistency_loss(dynamic_proj, gt_proj) * step_mask
        step_reward = reward_loss(value_prefix, target_reward, config) * step_mask
        step_value = value_loss(values, target_search_value, config) * step_mask
        step_policy, step_entropy = continuous_policy_loss(
            policy,
            best_action,
            entropy_rng=step_rng,
        )
        step_policy = step_policy * step_mask * policy_mask
        step_entropy = step_entropy * step_mask * policy_mask

        next_state = apply_half_gradient(next_state)
        return (next_state, next_reward_hidden), {
            "consistency": step_consistency,
            "reward": step_reward,
            "value": step_value,
            "policy": step_policy,
            "entropy": step_entropy,
        }

    step_actions = actions[:, :unroll_steps]
    step_best_actions = jnp.concatenate([actions[:, 1:], actions[:, -1:]], axis=1)
    step_rewards = rewards[:, :unroll_steps]
    step_search_values = search_values[:, 1 : unroll_steps + 1]
    step_next_obs = observations[:, 1 : unroll_steps + 1]
    step_masks = jnp.logical_not(dones[:, :unroll_steps]).astype(jnp.float32)
    policy_valid = (jnp.arange(unroll_steps) + 1) < unroll_steps
    step_policy_masks = policy_valid.astype(jnp.float32)
    step_rngs = jax.random.split(unroll_rng, unroll_steps)

    _, step_losses = jax.lax.scan(
        step_fn,
        (state, reward_hidden),
        (
            step_actions,
            step_rewards,
            step_search_values,
            step_next_obs,
            step_masks,
            step_policy_masks,
            step_best_actions,
            step_rngs,
        ),
    )

    consistency_loss_total = consistency_loss_total + jnp.sum(step_losses["consistency"], axis=0)
    reward_loss_total = reward_loss_total + jnp.sum(step_losses["reward"], axis=0)
    value_loss_total = value_loss_total + jnp.sum(step_losses["value"], axis=0)
    policy_loss_total = policy_loss_total + jnp.sum(step_losses["policy"], axis=0)
    entropy_total = entropy_total + jnp.sum(step_losses["entropy"], axis=0)

    total = (
        reward_loss_total * config.reward_loss_coeff
        + value_loss_total * config.value_loss_coeff
        + policy_loss_total * config.policy_loss_coeff
        + consistency_loss_total * config.consistency_coeff
        - entropy_total * config.entropy_coeff
    )
    loss = jnp.mean(total) / float(unroll_steps)

    metrics = {
        "loss": loss,
        "reward_loss": jnp.mean(reward_loss_total),
        "value_loss": jnp.mean(value_loss_total),
        "policy_loss": jnp.mean(policy_loss_total),
        "consistency_loss": jnp.mean(consistency_loss_total),
        "entropy": jnp.mean(entropy_total),
    }
    return loss, metrics


class EfficientZeroLearner(Learner):
    """Sample unroll windows and update the EfficientZero world model."""

    def __init__(self, context: ComponentContext) -> None:
        if not isinstance(context.config, EfficientZeroConfig):
            raise TypeError(
                "EfficientZeroLearner requires EfficientZeroConfig, "
                f"got {type(context.config)!r}."
            )
        if not isinstance(context.world_model, EfficientZeroWorldModel):
            raise TypeError(
                "EfficientZeroLearner requires EfficientZeroWorldModel, "
                f"got {type(context.world_model)!r}."
            )
        if context.planner is None:
            raise RuntimeError("EfficientZeroLearner requires a planner in the build context.")

        self.backend = context.backend
        self.config: EfficientZeroConfig = context.config
        self.world_model: EfficientZeroWorldModel = context.world_model
        self.planner = context.planner
        self.model = self.world_model.model
        self.params: Params = self.world_model.params

        if self.config.policy_distribution == "discrete":
            raise NotImplementedError(
                "EfficientZeroLearner discrete policy losses are not implemented yet."
            )

        self._rng_key = self.backend.random_key(self.config.seed + 2)
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        self._opt_state = self._optimizer.init(self.params)
        self._update = jax.jit(
            partial(
                _optimizer_step,
                model=self.model,
                config=self.config,
                optimizer=self._optimizer,
            )
        )

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        batch = replay_buffer.sample(self.config.batch_size)
        arrays = _batch_to_arrays(batch)
        self._rng_key, step_key = jax.random.split(self._rng_key)
        self.params, self._opt_state, metrics = self._update(
            self.params,
            self._opt_state,
            arrays,
            step_key,
        )
        self._sync_params()
        return {key: float(value) for key, value in metrics.items()}


    def _sync_params(self) -> None:
        self.world_model.params = self.params
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.params = self.params


def _optimizer_step(
    params: Params,
    opt_state: optax.OptState,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    *,
    model: EfficientZeroNetwork,
    config: EfficientZeroConfig,
    optimizer: optax.GradientTransformation,
) -> tuple[Params, optax.OptState, dict[str, jnp.ndarray]]:
    loss_fn = partial(
        _loss_from_batch,
        model=model,
        config=config,
    )

    def objective(current_params: Params) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return loss_fn(current_params, batch, rng=rng)

    (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    metrics = {**metrics, "loss": loss}
    return new_params, new_opt_state, metrics


def build_efficient_zero_learner(context: ComponentContext) -> EfficientZeroLearner:
    return EfficientZeroLearner(context)
