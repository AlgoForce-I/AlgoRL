"""EfficientZero learner with HyperCEZ sample-efficiency features."""

from __future__ import annotations

import copy
from functools import partial
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.learners.efficientzero.reanalyze import (
    batch_initial_values,
    effective_reanalyze_search_batch_size,
    mcts_temperature,
    reanalyze_policy_batch,
)
from algorl.buffers.efficientzero.targets import (
    prepare_bootstrapped_batch_values,
    prepare_gae_batch_values,
)
from algorl.backends.jax.nn.efficientzero.obs_norm import update_representation_obs_stats
from algorl.backends.jax.nn.efficientzero.losses import (
    _reduce_value_logits,
    apply_half_gradient,
    continuous_policy_loss,
    cosine_consistency_loss,
    reward_loss,
    value_loss,
)
from algorl.backends.jax.nn.efficientzero.model import EfficientZero as EfficientZeroNetwork
from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.planners.efficientzero import EfficientZeroPlanner
from algorl.backends.jax.world_models.efficientzero import EfficientZeroWorldModel
from algorl.buffers.efficientzero import EfficientZeroReplayBuffer
from algorl.core.component_context import ComponentContext
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch


def _batch_to_arrays(batch: Batch) -> dict[str, jnp.ndarray]:
    required = (
        "observations",
        "actions",
        "rewards",
        "policy_targets",
        "value_targets",
        "policy_candidates",
        "best_actions",
        "dones",
        "masks",
        "weights",
        "indices",
        "policy_masks",
    )
    data = batch.data
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"EfficientZero learner expected batch keys {missing}.")

    arrays: dict[str, jnp.ndarray] = {}
    for key in required:
        if key == "dones":
            arrays[key] = jnp.asarray(data[key]).astype(jnp.bool_)
        elif key == "policy_candidates":
            arrays[key] = jnp.asarray(data[key], dtype=jnp.float32)
        elif key == "policy_masks":
            arrays[key] = jnp.asarray(data[key], dtype=jnp.float32)
        else:
            arrays[key] = jnp.asarray(data[key], dtype=jnp.float32)
    return arrays


def _time_major_for_scan(array: jnp.ndarray) -> jnp.ndarray:
    """``jax.lax.scan`` runs over axis 0; training tensors are stored ``[B, T, ...]``."""
    if array.ndim <= 1:
        return array
    return jnp.swapaxes(array, 0, 1)


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
    value_targets = batch["value_targets"]
    policy_targets = batch["policy_targets"]
    policy_candidates = batch["policy_candidates"]
    best_actions = batch["best_actions"]
    masks = batch["masks"]
    weights = batch["weights"]
    dones = batch["dones"]

    unroll_steps = config.unroll_steps
    batch_size = observations.shape[0]
    action_dim = actions.shape[-1]

    rng, init_policy_rng, unroll_rng = jax.random.split(rng, 3)
    init_obs = observations[:, 0]

    state = model.do_representation(params, init_obs)
    values, policy = model.do_value_policy_prediction(params, state)

    value_loss_total = value_loss(values, value_targets[:, 0], config)
    policy_loss_total, entropy_total = continuous_policy_loss(
        policy,
        best_actions[:, 0],
        candidates=policy_candidates[:, 0],
        target_policy=policy_targets[:, 0],
        entropy_rng=init_policy_rng,
    )
    reward_loss_total = jnp.zeros((batch_size,), dtype=jnp.float32)
    consistency_loss_total = jnp.zeros((batch_size,), dtype=jnp.float32)
    pred_scalars = _reduce_value_logits(values, config)

    reward_hidden = None

    def step_fn(
        carry: tuple[jnp.ndarray, Any],
        step_inputs: tuple[jnp.ndarray, ...],
    ) -> tuple[tuple[jnp.ndarray, Any], dict[str, jnp.ndarray]]:
        state, reward_hidden = carry
        (
            step_index,
            action,
            target_reward,
            target_value,
            target_policy,
            candidates,
            best_action,
            next_obs,
            step_mask,
            step_rng,
        ) = step_inputs

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
        step_value = value_loss(values, target_value, config) * step_mask
        step_policy, step_entropy = continuous_policy_loss(
            policy,
            best_action,
            candidates=candidates,
            target_policy=target_policy,
            entropy_rng=step_rng,
        )
        step_policy = step_policy * step_mask
        step_entropy = step_entropy * step_mask

        next_state = apply_half_gradient(next_state)
        return (next_state, next_reward_hidden), {
            "consistency": step_consistency,
            "reward": step_reward,
            "value": step_value,
            "policy": step_policy,
            "entropy": step_entropy,
        }

    step_actions = _time_major_for_scan(actions[:, :unroll_steps])
    step_rewards = _time_major_for_scan(rewards[:, :unroll_steps])
    step_value_targets = _time_major_for_scan(value_targets[:, 1 : unroll_steps + 1])
    step_policy_targets = _time_major_for_scan(policy_targets[:, 1 : unroll_steps + 1])
    step_candidates = _time_major_for_scan(policy_candidates[:, 1 : unroll_steps + 1])
    step_best_actions = _time_major_for_scan(best_actions[:, 1 : unroll_steps + 1])
    step_next_obs = _time_major_for_scan(observations[:, 1 : unroll_steps + 1])
    step_masks = _time_major_for_scan(masks)
    step_rngs = jax.random.split(unroll_rng, unroll_steps)
    step_indices = jnp.arange(unroll_steps, dtype=jnp.int32)

    _, step_losses = jax.lax.scan(
        step_fn,
        (state, reward_hidden),
        (
            step_indices,
            step_actions,
            step_rewards,
            step_value_targets,
            step_policy_targets,
            step_candidates,
            step_best_actions,
            step_next_obs,
            step_masks,
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
    weighted = total * weights
    loss = jnp.sum(weighted) / (jnp.sum(weights) + 1e-8) / float(unroll_steps)
    priorities = jnp.abs(pred_scalars / float(unroll_steps) - value_targets[:, 0]) + config.min_prior

    metrics = {
        "loss": loss,
        "reward_loss": jnp.mean(reward_loss_total),
        "value_loss": jnp.mean(value_loss_total),
        "policy_loss": jnp.mean(policy_loss_total),
        "consistency_loss": jnp.mean(consistency_loss_total),
        "entropy": jnp.mean(entropy_total),
        "priorities": priorities,
    }
    return loss, metrics


class EfficientZeroLearner(Learner):
    """HyperCEZ-style EfficientZero learner with reanalyze and priority replay."""

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
        self._self_play_params: Params = copy.deepcopy(self.params)
        self._reanalyze_params: Params = copy.deepcopy(self.params)
        self._recent_reanalyze_params: Params = copy.deepcopy(self.params)
        self._train_steps = 0
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.self_play_params = self._self_play_params

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
        if not isinstance(replay_buffer, EfficientZeroReplayBuffer):
            raise TypeError(
                "EfficientZeroLearner requires EfficientZeroReplayBuffer, "
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
        )
        self.params = update_representation_obs_stats(
            self.params,
            np.asarray(arrays["observations"]),
        )
        self._propagate_obs_norm_stats()
        self._maybe_refresh_model_copies()
        self.params, self._opt_state, metrics = self._update(
            self.params,
            self._opt_state,
            arrays,
            step_key,
        )
        self._sync_params()
        replay_buffer.update_priorities(
            np.asarray(arrays["indices"]),
            np.asarray(metrics["priorities"]),
        )
        self._train_steps += 1
        return {
            key: float(value)
            for key, value in metrics.items()
            if key != "priorities"
        }

    def _priority_beta(self) -> float:
        if not self.config.use_priority:
            return 1.0
        total = max(1, self.config.total_training_steps)
        initial = self.config.priority_prob_beta
        progress = min(1.0, self._train_steps / total)
        return float(initial + (1.0 - initial) * progress)

    def _maybe_refresh_model_copies(self) -> None:
        self_play_interval = max(1, self.config.self_play_update_interval)
        if self._train_steps > 0 and self._train_steps % self_play_interval == 0:
            self._self_play_params = copy.deepcopy(self.params)
            if isinstance(self.planner, EfficientZeroPlanner):
                self.planner.self_play_params = self._self_play_params

        reanalyze_interval = max(1, self.config.reanalyze_update_interval)
        if self._train_steps > 0 and self._train_steps % reanalyze_interval == 0:
            self._reanalyze_params = copy.deepcopy(self._recent_reanalyze_params)
            self._recent_reanalyze_params = copy.deepcopy(self.params)

        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.params = self.params

    def _propagate_obs_norm_stats(self) -> None:
        """Keep online observation stats aligned across planner model copies."""
        source = self.params["representation_model"]
        for attr in ("_self_play_params", "_reanalyze_params", "_recent_reanalyze_params"):
            target = getattr(self, attr)
            rep = dict(target["representation_model"])
            for key in ("running_mean", "running_var", "running_count"):
                if key in source:
                    rep[key] = source[key]
            target["representation_model"] = rep
        if isinstance(self.planner, EfficientZeroPlanner):
            for params in (self.planner.params, self.planner.self_play_params):
                if params is None:
                    continue
                rep = dict(params["representation_model"])
                for key in ("running_mean", "running_var", "running_count"):
                    if key in source:
                        rep[key] = source[key]
                params["representation_model"] = rep

    def _sync_params(self) -> None:
        self.world_model.params = self.params
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.params = self.params


def _prepare_training_batch(
    batch: Batch,
    *,
    planner: Planner,
    config: EfficientZeroConfig,
    model: EfficientZeroNetwork,
    reanalyze_params: Params,
    trained_steps: int,
    total_transitions: int,
    rng_key: jax.Array,
    on_reanalyze_progress: Callable[[int, int], None] | None = None,
) -> dict[str, jnp.ndarray]:
    arrays = _batch_to_arrays(batch)
    observations = np.asarray(batch.data["observations"], dtype=np.float32)
    rewards = np.asarray(batch.data["rewards"], dtype=np.float32)
    sample_indices = np.asarray(batch.data["indices"], dtype=np.int32)

    def infer_values(obs: np.ndarray) -> np.ndarray:
        return batch_initial_values(
            model,
            reanalyze_params,
            obs,
            rng_key=rng_key,
            mini_batch_size=config.reanalyze_mini_batch_size,
        )

    if config.model_value_target == "bootstrapped":
        bootstrapped = prepare_bootstrapped_batch_values(
            observations,
            rewards,
            sample_indices,
            total_transitions=total_transitions,
            config=config,
            infer_values=infer_values,
        )
    else:
        bootstrapped = prepare_gae_batch_values(
            observations,
            rewards,
            sample_indices,
            total_transitions=total_transitions,
            config=config,
            infer_values=infer_values,
        )

    arrays["value_targets"] = jnp.asarray(bootstrapped, dtype=jnp.float32)
    arrays["search_values"] = jnp.asarray(batch.data["search_values"], dtype=jnp.float32)

    reanalyze_count = int(config.batch_size * config.reanalyze_ratio)
    policy_masks = np.asarray(batch.data.get("policy_masks", np.ones(observations.shape[:2])), dtype=np.float32)

    if reanalyze_count > 0 and isinstance(planner, EfficientZeroPlanner):
        old_params = planner.params
        planner.params = reanalyze_params
        try:
            temperature = mcts_temperature(config, trained_steps)
            (
                policy_targets,
                search_values,
                policy_candidates,
                best_actions,
                reanalyzed_masks,
            ) = reanalyze_policy_batch(
                planner,
                observations,
                reanalyze_count=reanalyze_count,
                temperature=temperature,
                search_batch_size=effective_reanalyze_search_batch_size(config),
                on_progress=on_reanalyze_progress,
            )
        finally:
            planner.params = old_params

        arrays["policy_targets"] = jnp.asarray(
            _merge_reanalyze(arrays["policy_targets"], policy_targets, reanalyze_count),
            dtype=jnp.float32,
        )
        arrays["search_values"] = jnp.asarray(
            _merge_reanalyze_1d(arrays["search_values"], search_values, reanalyze_count),
            dtype=jnp.float32,
        )
        arrays["policy_candidates"] = jnp.asarray(
            _merge_reanalyze(arrays["policy_candidates"], policy_candidates, reanalyze_count),
            dtype=jnp.float32,
        )
        arrays["best_actions"] = jnp.asarray(
            _merge_reanalyze(arrays["best_actions"], best_actions, reanalyze_count),
            dtype=jnp.float32,
        )
        policy_masks = _merge_reanalyze(policy_masks, reanalyzed_masks, reanalyze_count)

    if config.value_target == "search":
        arrays["value_targets"] = arrays["search_values"]
    elif config.value_target == "max":
        arrays["value_targets"] = jnp.maximum(arrays["value_targets"], arrays["search_values"])
    elif (
        config.value_target == "mixed"
        and trained_steps >= config.start_use_mix_training_steps
    ):
        mix_masks = jnp.asarray(batch.data["mix_masks"], dtype=jnp.float32)
        arrays["value_targets"] = (
            arrays["value_targets"] * mix_masks
            + arrays["search_values"] * (1.0 - mix_masks)
        )

    arrays["policy_masks"] = jnp.asarray(policy_masks, dtype=jnp.float32)
    return arrays


def _merge_reanalyze(
    original: jnp.ndarray | np.ndarray,
    refreshed: np.ndarray,
    count: int,
) -> np.ndarray:
    merged = np.asarray(original, dtype=np.float32).copy()
    merged[:count] = refreshed[:count]
    return merged


def _merge_reanalyze_1d(
    original: jnp.ndarray | np.ndarray,
    refreshed: np.ndarray,
    count: int,
) -> np.ndarray:
    merged = np.asarray(original, dtype=np.float32).copy()
    merged[:count] = refreshed[:count]
    return merged


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
