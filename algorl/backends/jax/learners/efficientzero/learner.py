"""EfficientZero-V2 learner."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.learners.efficientzero.reanalyze import (
    ValueInferenceFn,
    batch_initial_values,
    make_batched_value_inference,
    mcts_temperature,
    reanalyze_fused_policy_batches,
    reanalyze_policy_batch,
    resolve_reanalyze_search_width,
)
from algorl.buffers.efficientzero.targets import (
    prepare_bootstrapped_batch_values,
    prepare_gae_batch_values,
)
from algorl.backends.jax.nn.efficientzero.obs_norm import (
    with_representation_obs_stats,
    compute_tentative_obs_stats_jax,
)
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
from algorl.backends.jax.planners.efficientzero.planner import continuous_search_config_from_agent
from algorl.backends.jax.world_models.efficientzero import EfficientZeroWorldModel
from algorl.buffers.efficientzero import EfficientZeroReplayBuffer
from algorl.core.component_context import ComponentContext
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch
from algorl.envs.training_env import TrainingEnv


@dataclass(frozen=True)
class TrainingBurstSpec:
    """Fixed tensor shapes for a compiled gradient burst."""

    burst_steps: int
    batch_size: int
    window: int
    unroll_steps: int
    obs_dim: int
    action_dim: int
    policy_width: int


def training_burst_spec(config: EfficientZeroConfig, env: TrainingEnv) -> TrainingBurstSpec:
    """Derive stable burst shapes from config and the training env."""
    import gymnasium as gym

    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        obs_dim = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, gym.spaces.Discrete):
        obs_dim = int(obs_space.n)
    else:
        raise TypeError(f"Unsupported observation space for burst compile: {type(obs_space)!r}")

    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        action_dim = int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        action_dim = 1
    else:
        raise TypeError(f"Unsupported action space for burst compile: {type(action_space)!r}")

    compile_steps = config.burst_compile_steps
    if compile_steps is None:
        compile_steps = max(1, int(config.gradient_steps_per_rollout or 1))
    search_cfg = continuous_search_config_from_agent(config)
    return TrainingBurstSpec(
        burst_steps=max(1, int(compile_steps)),
        batch_size=config.batch_size,
        window=config.unroll_steps + 1,
        unroll_steps=config.unroll_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_width=search_cfg.num_sampled_actions,
    )


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
        # Halve gradients on the unrolled state (reward, value, policy, projection, dynamics).
        next_state = apply_half_gradient(next_state)
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
    # EfficientZero-V2: ``(weights * loss).mean()`` with a 1/unroll_steps gradient hook.
    loss = jnp.mean(weighted) / float(unroll_steps)

    # EfficientZero-V2 fresh priority: |min-ensemble value - target| on the first step,
    # clipped to [0, 1e5] for continuous control.
    pred_priority_values = pred_scalars
    if pred_priority_values.ndim == 2:
        pred_priority_values = jnp.min(pred_priority_values, axis=0)
    priority_targets = value_targets[:, 0]
    if config.clip_inference_values:
        pred_priority_values = jnp.clip(pred_priority_values, 0.0, 1e5)
        priority_targets = jnp.clip(priority_targets, 0.0, 1e5)
    priorities = jnp.abs(pred_priority_values - priority_targets) + config.min_prior
    priorities = jnp.nan_to_num(
        priorities,
        nan=config.min_prior,
        posinf=1e5 + config.min_prior,
        neginf=config.min_prior,
    )
    search_values = batch.get("search_values")
    search_target_gap = jnp.zeros((), dtype=jnp.float32)
    if search_values is not None:
        search_target_gap = jnp.mean(jnp.abs(search_values[:, 0] - value_targets[:, 0]))

    metrics = {
        "loss": loss,
        "reward_loss": jnp.mean(reward_loss_total),
        "value_loss": jnp.mean(value_loss_total),
        "policy_loss": jnp.mean(policy_loss_total),
        "consistency_loss": jnp.mean(consistency_loss_total),
        "entropy": jnp.mean(entropy_total),
        "value_pred_mae": jnp.mean(jnp.abs(pred_priority_values - priority_targets)),
        "value_target_mean": jnp.mean(priority_targets),
        "value_pred_mean": jnp.mean(pred_priority_values),
        "search_target_gap": search_target_gap,
        "priorities": priorities,
    }
    return loss, metrics


def _extract_obs_running_count(params: Params) -> int:
    rep = params.get("representation_model", {})
    count = rep.get("running_count")
    if count is None:
        return 1000
    return int(np.asarray(count, dtype=np.int64))


def _strip_obs_running_count(params: Params) -> Params:
    rep = dict(params["representation_model"])
    if "running_count" in rep:
        rep = {key: value for key, value in rep.items() if key != "running_count"}
    return {**params, "representation_model": rep}


class EfficientZeroLearner(Learner):
    """EfficientZero-V2 EfficientZero learner with reanalyze and priority replay."""

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
        self._obs_running_count = _extract_obs_running_count(self.world_model.params)
        self.params = _strip_obs_running_count(self.world_model.params)
        self.world_model.params = self.params
        self._self_play_params = _strip_obs_running_count(copy.deepcopy(self.params))
        self._reanalyze_params = _strip_obs_running_count(copy.deepcopy(self.params))
        self._recent_reanalyze_params = _strip_obs_running_count(copy.deepcopy(self.params))
        self._train_steps = 0
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.self_play_params = self._self_play_params

        if self.config.policy_distribution == "discrete":
            raise NotImplementedError(
                "EfficientZeroLearner discrete policy losses are not implemented yet."
            )
        if self.config.value_prefix:
            raise NotImplementedError(
                "value_prefix=True (reward-LSTM value prefix) is not "
                "implemented: the LSTM hidden state is not carried through MCTS "
                "or the learner unroll. Use value_prefix=False for vector-control."
            )

        self._rng_key = self.backend.random_key(self.config.seed + 2)
        # torch ``Adam(weight_decay=...)`` adds L2 to gradients before the Adam
        # update, so decayed weights go in front of the Adam transform.
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.add_decayed_weights(self.config.weight_decay),
            optax.adam(self.config.learning_rate),
        )
        self._opt_state = self._optimizer.init(self.params)
        self._burst_spec = training_burst_spec(self.config, context.env)
        self._value_infer_fn = make_batched_value_inference(self.model)
        # Resolve once so every reanalyze call compiles to a single width.
        self._reanalyze_search_width = resolve_reanalyze_search_width(
            self.config,
            action_dim=self._burst_spec.action_dim,
        )
        self._update = jax.jit(
            partial(
                _optimizer_step,
                model=self.model,
                config=self.config,
                optimizer=self._optimizer,
            )
        )
        self._burst_update = jax.jit(
            partial(
                _burst_optimizer_scan,
                model=self.model,
                config=self.config,
                optimizer=self._optimizer,
            )
        )
        self._warmup_burst_compile()

    def train_step(
        self,
        replay_buffer: ReplayBuffer,
        *,
        skip_reanalyze: bool = False,
    ) -> dict[str, float]:
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
            skip_reanalyze=skip_reanalyze,
            value_infer_fn=self._value_infer_fn,
            reanalyze_search_width=self._reanalyze_search_width,
        )
        self._maybe_refresh_model_copies()
        self.params, self._opt_state, self._obs_running_count, metrics = self._update(
            self.params,
            self._opt_state,
            jnp.asarray(self._obs_running_count, dtype=jnp.int32),
            arrays,
            step_key,
            jnp.asarray(self._learning_rate_scale(), dtype=jnp.float32),
        )
        self._obs_running_count = int(np.asarray(self._obs_running_count))
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

    def train_burst(
        self,
        replay_buffer: ReplayBuffer,
        steps: int,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, float]:
        """Run several gradient updates in compiled sub-bursts.

        Each sub-burst (``burst_compile_steps`` gradient steps) samples the
        buffer, computes value targets, and runs fused reanalyze *after* the
        previous sub-burst's parameter and priority updates. Freezing all of
        this across a long burst (as one big snapshot) measurably hurts sample
        efficiency versus the sequential loop, since the last gradient steps
        would train toward targets and priorities hundreds of steps stale.
        """
        if steps <= 0:
            return {}
        if not isinstance(replay_buffer, EfficientZeroReplayBuffer):
            raise TypeError(
                "EfficientZeroLearner requires EfficientZeroReplayBuffer, "
                f"got {type(replay_buffer)!r}."
            )

        metrics: dict[str, float] = {}
        completed = 0
        scan_width = self._burst_spec.burst_steps
        while completed < steps:
            chunk_steps = min(steps - completed, scan_width)
            metrics = self._train_burst_chunk(
                replay_buffer,
                chunk_steps,
                on_progress=on_progress,
                completed_steps=completed,
                total_steps=steps,
            )
            completed += chunk_steps
        return metrics

    def _train_burst_chunk(
        self,
        replay_buffer: EfficientZeroReplayBuffer,
        steps: int,
        *,
        on_progress: Callable[[int, int], None] | None,
        completed_steps: int,
        total_steps: int,
    ) -> dict[str, float]:
        """Sample, reanalyze, and run one compiled optimizer scan of ``steps`` updates."""
        beta = self._priority_beta()
        start_step = self._train_steps
        batches: list[Batch] = []
        for offset in range(steps):
            batches.append(
                replay_buffer.sample(
                    self.config.batch_size,
                    beta=beta,
                    trained_steps=start_step + offset,
                )
            )

        self._rng_key, burst_key = jax.random.split(self._rng_key)
        step_keys = jax.random.split(burst_key, steps)
        prepared_batches: list[dict[str, jnp.ndarray]] = []
        observation_windows: list[np.ndarray] = []
        window = self.config.unroll_steps + 1
        reanalyze_count = int(self.config.batch_size * self.config.reanalyze_ratio)

        for offset, batch in enumerate(batches):
            trained_steps = start_step + offset
            arrays = _prepare_training_batch(
                batch,
                planner=self.planner,
                config=self.config,
                model=self.model,
                reanalyze_params=self._reanalyze_params,
                trained_steps=trained_steps,
                total_transitions=replay_buffer.total_transitions,
                rng_key=step_keys[offset],
                skip_reanalyze=True,
                value_infer_fn=self._value_infer_fn,
            )
            prepared_batches.append(arrays)
            observation_windows.append(
                np.asarray(batch.data["observations"], dtype=np.float32)[:, :window]
            )

        if (
            reanalyze_count > 0
            and isinstance(self.planner, EfficientZeroPlanner)
            and observation_windows
        ):
            temperature = mcts_temperature(self.config, start_step)
            reanalyze_outputs = reanalyze_fused_policy_batches(
                self.planner,
                observation_windows,
                params=self._reanalyze_params,
                reanalyze_count=reanalyze_count,
                temperature=temperature,
                search_batch_size=self._reanalyze_search_width,
                on_progress=getattr(self, "_on_reanalyze_progress", None),
            )
            for offset, refreshed in enumerate(reanalyze_outputs):
                prepared_batches[offset] = _apply_reanalyze_outputs(
                    prepared_batches[offset],
                    refreshed,
                    reanalyze_count,
                )
                prepared_batches[offset] = _finalize_value_targets(
                    prepared_batches[offset],
                    batches[offset],
                    config=self.config,
                    trained_steps=start_step + offset,
                )
        else:
            for offset in range(steps):
                prepared_batches[offset] = _finalize_value_targets(
                    prepared_batches[offset],
                    batches[offset],
                    config=self.config,
                    trained_steps=start_step + offset,
                )

        return self._run_burst_scan_chunk(
            replay_buffer,
            prepared_batches=prepared_batches,
            step_keys=step_keys,
            start_step=start_step,
            chunk_steps=steps,
            on_progress=on_progress,
            completed_steps=completed_steps,
            total_steps=total_steps,
        )

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
        self.params, self._opt_state, self._obs_running_count, burst_metrics = self._burst_update(
            self.params,
            self._opt_state,
            jnp.asarray(self._obs_running_count, dtype=jnp.int32),
            stacked_batch,
            step_keys,
            lr_scales,
            active_mask,
        )
        self._obs_running_count = int(np.asarray(self._obs_running_count))
        self._train_steps = start_step + chunk_steps
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

    def _warmup_burst_compile(self) -> None:
        """Compile the burst scan once with the fixed training shapes."""
        spec = self._burst_spec
        if spec.burst_steps <= 1:
            return
        self._rng_key, warmup_key = jax.random.split(self._rng_key)
        dummy_batches = [
            _dummy_training_batch(spec, warmup_key) for _ in range(spec.burst_steps)
        ]
        stacked_batch = _stack_training_batches(dummy_batches)
        step_keys = jax.random.split(warmup_key, spec.burst_steps)
        lr_scales = jnp.ones((spec.burst_steps,), dtype=jnp.float32)
        active_mask = jnp.ones((spec.burst_steps,), dtype=jnp.float32)
        _, _, _, _ = self._burst_update(
            self.params,
            self._opt_state,
            jnp.asarray(self._obs_running_count, dtype=jnp.int32),
            stacked_batch,
            step_keys,
            lr_scales,
            active_mask,
        )

    def _learning_rate_scale_at(self, trained_steps: int) -> float:
        """Learning-rate scale for a specific global train step."""
        total = max(1, self.config.total_training_steps)
        warm_steps = int(total * self.config.lr_warm_up)
        if warm_steps > 0 and trained_steps < warm_steps:
            return trained_steps / warm_steps
        decay_steps = max(1, self.config.lr_decay_steps)
        exponent = (trained_steps - warm_steps) // decay_steps
        return float(self.config.lr_decay_rate**exponent)

    def _learning_rate_scale(self) -> float:
        """Linear LR warmup, then step decay."""
        return self._learning_rate_scale_at(self._train_steps)

    def _priority_beta(self) -> float:
        if not self.config.use_priority:
            return 1.0
        total = max(1, self.config.total_training_steps)
        initial = self.config.priority_prob_beta
        progress = min(1.0, self._train_steps / total)
        return float(initial + (1.0 - initial) * progress)

    def sync_self_play_for_rollout(self) -> None:
        """Copy learner weights to self-play MCTS before collecting rollout data.

        Used when ``config.sync_self_play_before_rollout`` is True. EZ-V2 default
        behavior leaves this off and refreshes only every
        ``self_play_update_interval`` train steps.
        """
        self._propagate_obs_norm_stats()
        self._self_play_params = copy.deepcopy(self.params)
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.self_play_params = self._self_play_params
            self.planner.params = self.params

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
            for key in ("running_mean", "running_var"):
                if key in source:
                    rep[key] = source[key]
            target["representation_model"] = rep
        if isinstance(self.planner, EfficientZeroPlanner):
            for params in (self.planner.params, self.planner.self_play_params):
                if params is None:
                    continue
                rep = dict(params["representation_model"])
                for key in ("running_mean", "running_var"):
                    if key in source:
                        rep[key] = source[key]
                params["representation_model"] = rep

    def _sync_params(self) -> None:
        self.world_model.params = self.params
        if isinstance(self.planner, EfficientZeroPlanner):
            self.planner.params = self.params

    def checkpoint_state(self) -> dict[str, Any]:
        """Structured learner state for multi-file checkpoints."""
        from algorl.backends.jax.learners.efficientzero.checkpoint import (
            efficient_zero_checkpoint_state,
        )

        return {
            "meta": {
                "train_steps": int(self._train_steps),
                "obs_running_count": int(self._obs_running_count),
            },
            "data": efficient_zero_checkpoint_state(self),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`checkpoint_state` (in-memory helper)."""
        from algorl.backends.jax.learners.efficientzero.checkpoint import (
            apply_efficient_zero_learner_state,
        )

        apply_efficient_zero_learner_state(
            self,
            meta=state["meta"],
            data=state["data"],
        )

    def save(self, directory: str | Path) -> None:
        """Persist learner weights, optimizer state, and RNGs to ``directory``."""
        from algorl.backends.jax.learners.efficientzero.checkpoint import (
            save_efficient_zero_learner,
        )

        save_efficient_zero_learner(self, directory)

    def load(self, directory: str | Path) -> None:
        """Restore learner state written by :meth:`save`."""
        from algorl.backends.jax.learners.efficientzero.checkpoint import (
            load_efficient_zero_learner,
        )

        load_efficient_zero_learner(self, directory)


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
    skip_reanalyze: bool = False,
    value_infer_fn: ValueInferenceFn | None = None,
    reanalyze_search_width: int | None = None,
) -> dict[str, jnp.ndarray]:
    arrays = _batch_to_arrays(batch)
    window = config.unroll_steps + 1
    observations = np.asarray(batch.data["observations"], dtype=np.float32)
    rewards = np.asarray(batch.data["rewards"], dtype=np.float32)
    sample_indices = np.asarray(batch.data["indices"], dtype=np.int32)
    batch_size = observations.shape[0]
    valid_lengths = np.asarray(
        batch.data.get("valid_lengths", np.full((batch_size,), observations.shape[1], dtype=np.int32)),
        dtype=np.int32,
    )
    bootstrap_limits = np.asarray(
        batch.data.get("bootstrap_limits", valid_lengths),
        dtype=np.int32,
    )

    def infer_values(obs: np.ndarray) -> np.ndarray:
        return batch_initial_values(
            model,
            reanalyze_params,
            obs,
            rng_key=rng_key,
            mini_batch_size=config.reanalyze_mini_batch_size,
            infer_fn=value_infer_fn,
        )

    if config.model_value_target == "bootstrapped":
        bootstrapped = prepare_bootstrapped_batch_values(
            observations,
            rewards,
            sample_indices,
            valid_lengths=valid_lengths,
            bootstrap_limits=bootstrap_limits,
            total_transitions=total_transitions,
            config=config,
            infer_values=infer_values,
        )
    else:
        bootstrapped = prepare_gae_batch_values(
            observations,
            rewards,
            sample_indices,
            valid_lengths=valid_lengths,
            bootstrap_limits=bootstrap_limits,
            total_transitions=total_transitions,
            config=config,
            infer_values=infer_values,
        )

    # The unroll losses only consume the training window; the extended tail
    # exists solely for full-horizon value targets above.
    arrays["observations"] = arrays["observations"][:, :window]
    arrays["value_targets"] = jnp.asarray(bootstrapped, dtype=jnp.float32)
    arrays["search_values"] = jnp.asarray(batch.data["search_values"], dtype=jnp.float32)

    reanalyze_count = 0 if skip_reanalyze else int(config.batch_size * config.reanalyze_ratio)

    if reanalyze_count > 0 and isinstance(planner, EfficientZeroPlanner):
        temperature = mcts_temperature(config, trained_steps)
        if reanalyze_search_width is None:
            reanalyze_search_width = resolve_reanalyze_search_width(config)
        policy_targets, search_values, policy_candidates, best_actions = reanalyze_policy_batch(
            planner,
            observations[:, :window],
            params=reanalyze_params,
            reanalyze_count=reanalyze_count,
            temperature=temperature,
            search_batch_size=reanalyze_search_width,
            on_progress=on_reanalyze_progress,
        )
        arrays = _apply_reanalyze_outputs(
            arrays,
            (policy_targets, search_values, policy_candidates, best_actions),
            reanalyze_count,
        )

    return _finalize_value_targets(arrays, batch, config=config, trained_steps=trained_steps)


def _apply_reanalyze_outputs(
    arrays: dict[str, jnp.ndarray],
    refreshed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    reanalyze_count: int,
) -> dict[str, jnp.ndarray]:
    policy_targets, search_values, policy_candidates, best_actions = refreshed
    arrays = dict(arrays)
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
    return arrays


def _finalize_value_targets(
    arrays: dict[str, jnp.ndarray],
    batch: Batch,
    *,
    config: EfficientZeroConfig,
    trained_steps: int,
) -> dict[str, jnp.ndarray]:
    arrays = dict(arrays)
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
    return arrays


def _pad_training_batch_to_spec(
    batch: dict[str, jnp.ndarray],
    spec: TrainingBurstSpec,
) -> dict[str, jnp.ndarray]:
    padded = dict(batch)
    padded["policy_targets"] = _pad_axis(
        jnp.asarray(batch["policy_targets"], dtype=jnp.float32),
        axis=2,
        size=spec.policy_width,
    )
    padded["policy_candidates"] = _pad_axis(
        jnp.asarray(batch["policy_candidates"], dtype=jnp.float32),
        axis=2,
        size=spec.policy_width,
    )
    padded["best_actions"] = jnp.asarray(batch["best_actions"], dtype=jnp.float32)
    if int(padded["best_actions"].shape[-1]) < spec.action_dim:
        padded["best_actions"] = _pad_axis(padded["best_actions"], axis=2, size=spec.action_dim)
    return padded


def _dummy_training_batch(spec: TrainingBurstSpec, rng_key: jax.Array) -> dict[str, jnp.ndarray]:
    batch_size = spec.batch_size
    return {
        "observations": jnp.zeros((batch_size, spec.window, spec.obs_dim), dtype=jnp.float32),
        "actions": jnp.zeros((batch_size, spec.unroll_steps, spec.action_dim), dtype=jnp.float32),
        "rewards": jnp.zeros((batch_size, spec.unroll_steps), dtype=jnp.float32),
        "policy_targets": jnp.zeros(
            (batch_size, spec.window, spec.policy_width),
            dtype=jnp.float32,
        ),
        "value_targets": jnp.zeros((batch_size, spec.window), dtype=jnp.float32),
        "policy_candidates": jnp.zeros(
            (batch_size, spec.window, spec.policy_width, spec.action_dim),
            dtype=jnp.float32,
        ),
        "best_actions": jnp.zeros((batch_size, spec.window, spec.action_dim), dtype=jnp.float32),
        "dones": jnp.zeros((batch_size, spec.unroll_steps), dtype=jnp.bool_),
        "masks": jnp.ones((batch_size, spec.unroll_steps), dtype=jnp.float32),
        "weights": jnp.ones((batch_size,), dtype=jnp.float32),
        "indices": jnp.zeros((batch_size,), dtype=jnp.float32),
        "search_values": jnp.zeros((batch_size, spec.window), dtype=jnp.float32),
    }


def _pad_burst_batches(
    batches: list[dict[str, jnp.ndarray]],
    spec: TrainingBurstSpec,
    *,
    active_steps: int,
) -> tuple[list[dict[str, jnp.ndarray]], jnp.ndarray]:
    padded = [_pad_training_batch_to_spec(batch, spec) for batch in batches]
    if not padded:
        padded = [_dummy_training_batch(spec, jax.random.PRNGKey(0))]
    while len(padded) < spec.burst_steps:
        padded.append(padded[-1])
    if len(padded) > spec.burst_steps:
        padded = padded[: spec.burst_steps]
    active_mask = np.zeros((spec.burst_steps,), dtype=np.float32)
    active_mask[: min(active_steps, spec.burst_steps)] = 1.0
    return padded, jnp.asarray(active_mask, dtype=jnp.float32)


def _pad_burst_scan_inputs(
    step_keys: jax.Array,
    lr_scales: list[float],
    *,
    spec: TrainingBurstSpec,
) -> tuple[jax.Array, jnp.ndarray]:
    keys = step_keys
    if int(keys.shape[0]) < spec.burst_steps:
        pad = jnp.repeat(keys[-1:], spec.burst_steps - int(keys.shape[0]), axis=0)
        keys = jnp.concatenate([keys, pad], axis=0)
    else:
        keys = keys[: spec.burst_steps]
    scales = list(lr_scales)
    while len(scales) < spec.burst_steps:
        scales.append(scales[-1] if scales else 1.0)
    return keys, jnp.asarray(scales[: spec.burst_steps], dtype=jnp.float32)


def _zeroed_step_metrics(batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    batch_size = int(batch["observations"].shape[0])
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return {
        "loss": zero,
        "reward_loss": zero,
        "value_loss": zero,
        "policy_loss": zero,
        "consistency_loss": zero,
        "entropy": zero,
        "value_pred_mae": zero,
        "value_target_mean": zero,
        "value_pred_mean": zero,
        "search_target_gap": zero,
        "priorities": jnp.zeros((batch_size,), dtype=jnp.float32),
    }


def _pad_axis(array: jnp.ndarray, *, axis: int, size: int) -> jnp.ndarray:
    current = int(array.shape[axis])
    if current >= size:
        return array
    pad_width = [(0, 0) for _ in range(array.ndim)]
    pad_width[axis] = (0, size - current)
    return jnp.pad(array, pad_width)


def _stack_training_batches(batches: list[dict[str, jnp.ndarray]]) -> dict[str, jnp.ndarray]:
    return jax.tree.map(lambda *items: jnp.stack(items, axis=0), *batches)


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


def _burst_optimizer_scan(
    params: Params,
    opt_state: optax.OptState,
    obs_count: jnp.ndarray,
    stacked_batch: dict[str, jnp.ndarray],
    rngs: jax.Array,
    lr_scales: jax.Array,
    active_mask: jax.Array,
    *,
    model: EfficientZeroNetwork,
    config: EfficientZeroConfig,
    optimizer: optax.GradientTransformation,
) -> tuple[Params, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]:
    def scan_step(
        carry: tuple[Params, optax.OptState, jnp.ndarray],
        inputs: tuple[dict[str, jnp.ndarray], jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[Params, optax.OptState, jnp.ndarray], dict[str, jnp.ndarray]]:
        current_params, current_opt_state, current_obs_count = carry
        batch, rng, lr_scale, active = inputs

        def run_step(
            _: None,
        ) -> tuple[tuple[Params, optax.OptState, jnp.ndarray], dict[str, jnp.ndarray]]:
            new_params, new_opt_state, new_obs_count, metrics = _optimizer_step(
                current_params,
                current_opt_state,
                current_obs_count,
                batch,
                rng,
                lr_scale,
                model=model,
                config=config,
                optimizer=optimizer,
            )
            return (new_params, new_opt_state, new_obs_count), metrics

        def skip_step(
            _: None,
        ) -> tuple[tuple[Params, optax.OptState, jnp.ndarray], dict[str, jnp.ndarray]]:
            return (current_params, current_opt_state, current_obs_count), _zeroed_step_metrics(
                batch
            )

        return jax.lax.cond(active > 0.0, run_step, skip_step, operand=None)

    scan_inputs = (stacked_batch, rngs, lr_scales, active_mask)
    (params, opt_state, obs_count), metrics = jax.lax.scan(
        scan_step, (params, opt_state, obs_count), scan_inputs
    )
    stacked_metrics = jax.tree.map(lambda leaf: jnp.asarray(leaf), metrics)
    return params, opt_state, obs_count, stacked_metrics


def _optimizer_step(
    params: Params,
    opt_state: optax.OptState,
    obs_count: jnp.ndarray,
    batch: dict[str, jnp.ndarray],
    rng: jax.Array,
    lr_scale: jnp.ndarray,
    *,
    model: EfficientZeroNetwork,
    config: EfficientZeroConfig,
    optimizer: optax.GradientTransformation,
) -> tuple[Params, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]:
    tentative_mean, tentative_var, tentative_count = compute_tentative_obs_stats_jax(
        params,
        batch["observations"],
        obs_count,
    )
    forward_params = with_representation_obs_stats(
        params,
        mean=tentative_mean,
        var=tentative_var,
    )

    loss_fn = partial(
        _loss_from_batch,
        model=model,
        config=config,
    )

    def objective(current_params: Params) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return loss_fn(current_params, batch, rng=rng)

    (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(forward_params)

    def merge_metrics() -> dict[str, jnp.ndarray]:
        merged = _zeroed_step_metrics(batch)
        for key in merged:
            if key in metrics:
                merged[key] = metrics[key]
        merged["loss"] = loss
        return merged

    def apply_step(_: None) -> tuple[Params, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]:
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        # EfficientZero-V2 mutates ``param_group['lr']`` per step; Adam updates scale
        # linearly in the learning rate, so scaling the final update is identical.
        updates = jax.tree.map(lambda update: update * lr_scale, updates)
        new_params = optax.apply_updates(params, updates)
        committed_params = with_representation_obs_stats(
            new_params,
            mean=tentative_mean,
            var=tentative_var,
        )
        return committed_params, new_opt_state, tentative_count, merge_metrics()

    def skip_step(_: None) -> tuple[Params, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]:
        return params, opt_state, obs_count, merge_metrics()

    return jax.lax.cond(jnp.isfinite(loss), apply_step, skip_step, None)


def build_efficient_zero_learner(context: ComponentContext) -> EfficientZeroLearner:
    if not isinstance(context.config, EfficientZeroConfig):
        if getattr(context.config, "require_implemented", True) is False:
            from algorl.backends.jax.learners import _StubLearner

            return _StubLearner("efficient_zero", context)  # type: ignore[return-value]
        raise TypeError(
            "EfficientZeroLearner requires EfficientZeroConfig, "
            f"got {type(context.config)!r}."
        )
    return EfficientZeroLearner(context)
