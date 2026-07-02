"""Shared environment interaction and training orchestration."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import BaseAgentConfig
from algorl.common.callbacks import Callback, CallbackList
from algorl.common.checkpoints import save_checkpoint
from algorl.common.episode_metrics import EpisodeMetricsTracker
from algorl.common.logger import Logger
from algorl.common.progress_bar import TqdmProgressBar
from algorl.common.tensorboard_logger import TensorboardLogger
from algorl.core.learner import Learner
from algorl.core.planner import BatchedPlanner, Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Action, Observation, Transition
from algorl.envs.jax_env import JaxRolloutBatch, PolicyFn
from algorl.envs.training_env import TrainingEnv


class TrainingLoop:
    """Collect environment data, train on a schedule, and emit metrics."""

    def __init__(
        self,
        *,
        env: TrainingEnv,
        planner: Planner,
        learner: Learner,
        replay_buffer: ReplayBuffer,
        config: BaseAgentConfig,
        callbacks: Callback | CallbackList | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.env = env
        self.planner = planner
        self.learner = learner
        self.replay_buffer = replay_buffer
        self.config = config
        self.callbacks = callbacks if isinstance(callbacks, CallbackList) else CallbackList(
            [callbacks] if callbacks is not None else []
        )
        self.logger = logger or Logger()
        self._episode_tracker = EpisodeMetricsTracker()
        self._key = jax.random.PRNGKey(config.seed)

    def run(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None = None,
        extra_step_info: dict[str, Any] | None = None,
        progress_bar: TqdmProgressBar | None = None,
    ) -> None:
        """Interact with the environment and call ``learner.train_step`` on schedule."""
        if progress_bar is not None:
            progress_bar.start(total_timesteps)
        try:
            if self.env.is_batched:
                self._run_batched(
                    total_timesteps,
                    checkpoint_path=checkpoint_path,
                    extra_step_info=extra_step_info,
                    progress_bar=progress_bar,
                )
                return
            self._run_sequential(
                total_timesteps,
                checkpoint_path=checkpoint_path,
                extra_step_info=extra_step_info,
                progress_bar=progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
            self._close_logger()

    def _run_sequential(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None,
        extra_step_info: dict[str, Any] | None,
        progress_bar: TqdmProgressBar | None,
    ) -> None:
        observation, reset_info = self.env.reset(seed=self.config.seed)
        self._episode_tracker.begin_episode(reset_info)
        metrics: dict[str, Any] = {}

        for step in range(total_timesteps):
            action = self._select_action(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(
                Transition(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    next_observation=next_observation,
                    done=done,
                    info=self._enrich_transition_info(dict(info)),
                )
            )

            episode_event = self._episode_tracker.observe_step(float(reward), done, dict(info))
            if episode_event is not None:
                metrics = self._record_episode(step, episode_event, metrics)

            if done:
                observation, reset_info = self.env.reset()
                self._episode_tracker.begin_episode(reset_info)
            else:
                observation = next_observation

            metrics = self._maybe_train(step, metrics)
            step_info = self._log_step(
                step,
                reward=reward,
                done=done,
                metrics=metrics,
                extra_step_info=extra_step_info,
            )
            self._maybe_checkpoint(step, metrics, checkpoint_path)
            if progress_bar is not None:
                progress_bar.update(step_info)

    def _run_batched(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None,
        extra_step_info: dict[str, Any] | None,
        progress_bar: TqdmProgressBar | None,
    ) -> None:
        chunk_size = max(1, self.config.jax_rollout_chunk)
        metrics: dict[str, Any] = {}
        steps_collected = 0
        step_counter = 0

        while steps_collected < total_timesteps:
            chunk_steps = min(chunk_size, total_timesteps - steps_collected)
            self._key, rollout_key = jax.random.split(self._key)
            search_results: list[Any] = []
            batch = self.env.collect_rollout(
                self._batched_policy(search_results),
                chunk_steps,
                key=rollout_key,
            )
            transitions = self._transitions_from_rollout(batch, search_results)
            for transition in transitions:
                self.replay_buffer.add(transition)
                steps_collected += 1
                episode_event = self._episode_tracker.observe_step(
                    float(transition.reward),
                    bool(transition.done),
                    dict(transition.info),
                )
                if episode_event is not None:
                    metrics = self._record_episode(step_counter, episode_event, metrics)
                    self._episode_tracker.begin_episode(dict(transition.info))
                metrics = self._maybe_train(step_counter, metrics)
                step_info = self._log_step(
                    step_counter,
                    reward=transition.reward,
                    done=transition.done,
                    metrics=metrics,
                    extra_step_info=extra_step_info,
                )
                self._maybe_checkpoint(step_counter, metrics, checkpoint_path)
                if progress_bar is not None:
                    progress_bar.update(step_info)
                step_counter += 1
                if steps_collected >= total_timesteps:
                    break

    def _batched_policy(self, search_results: list[Any] | None = None) -> PolicyFn:
        planner = self.planner

        if isinstance(planner, BatchedPlanner):
            def policy(observations: jnp.ndarray | np.ndarray, key: jnp.ndarray) -> jnp.ndarray:
                del key
                obs_array = np.asarray(observations, dtype=np.float32)
                obs_batch = [obs_array[lane] for lane in range(obs_array.shape[0])]
                result = planner.search_batch(obs_batch, deterministic=False)
                self.planner.last_result = result
                if search_results is not None:
                    search_results.append(result)
                return jnp.asarray(result.actions, dtype=jnp.float32)

            return policy

        def policy(observations: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
            del key
            actions = []
            for lane in range(observations.shape[0]):
                action = self._select_action(np.asarray(observations[lane], dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
            return jnp.asarray(actions, dtype=jnp.float32)

        return policy

    def _transitions_from_rollout(
        self,
        batch: JaxRolloutBatch,
        search_results: list[Any],
    ) -> list[Transition]:
        transitions: list[Transition] = []
        for step_idx in range(batch.num_steps):
            result = search_results[step_idx] if step_idx < len(search_results) else None
            for env_idx in range(batch.num_envs):
                info = self._search_info_from_result(result, env_idx)
                if batch.step_info is not None:
                    info = {**batch.step_info[step_idx][env_idx], **info}
                transitions.append(
                    Transition(
                        observation=batch.observation[step_idx, env_idx],
                        action=batch.action[step_idx, env_idx],
                        reward=float(batch.reward[step_idx, env_idx]),
                        next_observation=batch.next_observation[step_idx, env_idx],
                        done=bool(batch.done[step_idx, env_idx]),
                        info=info,
                    )
                )
        return transitions

    def _record_episode(
        self,
        step: int,
        event: object,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        from algorl.common.episode_metrics import EpisodeEndEvent

        if not isinstance(event, EpisodeEndEvent):
            return metrics
        if isinstance(self.logger, TensorboardLogger):
            episode_metrics = self.logger.record_episode(step, event)
        else:
            episode_metrics = self._episode_tracker.metrics_from_event(event)
        return {**metrics, **episode_metrics}

    def _close_logger(self) -> None:
        close = getattr(self.logger, "close", None)
        if callable(close):
            close()

    def _maybe_train(self, step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        if self._should_train(step):
            return self.learner.train_step(self.replay_buffer)
        return metrics

    def _log_step(
        self,
        step: int,
        *,
        reward: float,
        done: bool,
        metrics: dict[str, Any],
        extra_step_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prefixed_metrics = {
            (key if str(key).startswith("train/") else f"train/{key}"): value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        step_info = {"train/reward": float(reward), "done": done, **prefixed_metrics}
        if extra_step_info:
            step_info.update(extra_step_info)
        self.logger.record(step, step_info)
        self.callbacks.on_step(step, step_info)
        return step_info

    def _maybe_checkpoint(
        self,
        step: int,
        metrics: dict[str, Any],
        checkpoint_path: str | None,
    ) -> None:
        if (
            checkpoint_path is not None
            and self.config.checkpoint_freq is not None
            and step > 0
            and step % self.config.checkpoint_freq == 0
        ):
            save_checkpoint(
                checkpoint_path,
                {
                    "step": step,
                    "metrics": metrics,
                    "buffer_size": len(self.replay_buffer),
                },
            )

    def _select_action(self, observation: Observation) -> Action:
        return self.planner.search(observation, deterministic=False)

    def _enrich_transition_info(self, info: dict[str, object]) -> dict[str, object]:
        last_result = getattr(self.planner, "last_result", None)
        if last_result is None:
            return info
        enriched = dict(info)
        for key, value in self._search_info_from_result(last_result, 0).items():
            enriched.setdefault(key, value)
        return enriched

    def _search_info_from_result(self, result: Any, index: int) -> dict[str, object]:
        if result is None:
            return {}
        batch_size = int(getattr(result, "batch_size", 1))
        if not 0 <= index < batch_size:
            return {}
        info: dict[str, object] = {
            "policy_target": np.asarray(result.action_weights[index], dtype=np.float32),
            "search_value": float(result.root_values[index]),
            "root_candidates": np.asarray(result.root_candidates[index], dtype=np.float32),
            "best_action": np.asarray(result.actions[index], dtype=np.float32),
        }
        pred_values = getattr(result, "pred_values", None)
        if pred_values is not None:
            info["pred_value"] = float(pred_values[index])
        return info

    def _should_train(self, step: int) -> bool:
        if step < self.config.learning_starts:
            return False
        if self.config.train_freq <= 0 or step % self.config.train_freq != 0:
            return False
        return len(self.replay_buffer) >= self.config.batch_size
