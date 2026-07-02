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
from algorl.envs.jax_env import JaxRolloutBatch, PolicyFn, RolloutStepCallback
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
            bar_kwargs = {"mininterval": 0} if self.env.is_batched else {}
            progress_bar.start(total_timesteps, **bar_kwargs)
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

            metrics = self._maybe_train(step, metrics, progress_bar=progress_bar)
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
            rollout_progress = 0

            chunk_start_step = step_counter

            def on_rollout_step(env_steps: int, rollout_step_info: dict[str, Any]) -> None:
                nonlocal rollout_progress, metrics, step_counter
                metrics, rollout_progress, step_counter = self._consume_rollout_vector_step(
                    rollout_step_info,
                    env_steps=env_steps,
                    total_timesteps=total_timesteps,
                    steps_collected=steps_collected,
                    rollout_progress=rollout_progress,
                    step_counter=step_counter,
                    metrics=metrics,
                    extra_step_info=extra_step_info,
                    progress_bar=progress_bar,
                )

            batch = self.env.collect_rollout(
                self._batched_policy(search_results),
                chunk_steps,
                key=rollout_key,
                on_step=on_rollout_step,
            )
            transitions = self._transitions_from_rollout(batch, search_results)
            num_added = 0
            for transition in transitions:
                self.replay_buffer.add(transition)
                steps_collected += 1
                num_added += 1
                if steps_collected >= total_timesteps:
                    break

            if num_added == 0:
                continue
            chunk_end_step = chunk_start_step + num_added - 1
            for train_step in self._batched_gradient_steps(chunk_start_step, chunk_end_step):
                metrics = self._maybe_train(train_step, metrics, progress_bar=progress_bar)
                self._maybe_checkpoint(train_step, metrics, checkpoint_path)

    def _batched_gradient_steps(self, chunk_start: int, chunk_end: int) -> list[int]:
        """Return global env steps that should trigger ``train_step`` after one rollout chunk."""
        capped = self.config.gradient_steps_per_rollout
        if capped is None:
            return [
                step
                for step in range(chunk_start, chunk_end + 1)
                if self._should_train(step)
            ]
        if chunk_end < self.config.learning_starts:
            return []
        if len(self.replay_buffer) < self.config.batch_size:
            return []

        scheduled = [
            step
            for step in range(chunk_start, chunk_end + 1)
            if self._should_train(step)
        ]
        if not scheduled:
            scheduled = [chunk_end]
        if capped <= 0:
            return []
        if len(scheduled) > capped:
            scheduled = scheduled[-capped:]
        return scheduled

    def _consume_rollout_vector_step(
        self,
        rollout_step_info: dict[str, Any],
        *,
        env_steps: int,
        total_timesteps: int,
        steps_collected: int,
        rollout_progress: int,
        step_counter: int,
        metrics: dict[str, Any],
        extra_step_info: dict[str, Any] | None,
        progress_bar: TqdmProgressBar | None,
    ) -> tuple[dict[str, Any], int, int]:
        """Log per-env metrics during rollout so TensorBoard updates while MCTS runs."""
        rewards = rollout_step_info.get("rewards")
        dones = rollout_step_info.get("dones")
        infos = rollout_step_info.get("infos")
        if rewards is None or dones is None or infos is None:
            remaining = total_timesteps - steps_collected - rollout_progress
            increment = min(int(env_steps), remaining)
            if increment > 0 and progress_bar is not None:
                progress_bar.update({**rollout_step_info, "phase": "rollout"}, n=increment)
            return metrics, rollout_progress + increment, step_counter + increment

        reward_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
        done_array = np.asarray(dones, dtype=bool).reshape(-1)
        lane_infos = list(infos)
        processed = 0
        success_values: list[float] = []

        for lane in range(min(int(env_steps), reward_array.shape[0], len(lane_infos))):
            if steps_collected + rollout_progress >= total_timesteps:
                break
            info = dict(lane_infos[lane])
            reward = float(reward_array[lane])
            done = bool(done_array[lane])
            if "success" in info:
                success_values.append(float(info["success"]))

            episode_event = self._episode_tracker.observe_step(reward, done, info)
            if episode_event is not None:
                metrics = self._record_episode(step_counter, episode_event, metrics)
                self._episode_tracker.begin_episode(info)

            self._log_step(
                step_counter,
                reward=reward,
                done=done,
                metrics=metrics,
                extra_step_info=extra_step_info,
            )
            step_counter += 1
            rollout_progress += 1
            processed += 1

        if processed > 0:
            batched_metrics: dict[str, float] = {
                "train/batched/mean_reward": float(np.mean(reward_array[:processed])),
            }
            if success_values:
                batched_metrics["train/batched/success_frac"] = float(np.mean(success_values))
            task_name = rollout_step_info.get("task_name")
            if isinstance(task_name, str) and task_name:
                from algorl.common.episode_metrics import sanitize_task_name

                safe_name = sanitize_task_name(task_name)
                batched_metrics[f"train/batched/task/{safe_name}/mean_reward"] = batched_metrics[
                    "train/batched/mean_reward"
                ]
                if success_values:
                    batched_metrics[f"train/batched/task/{safe_name}/success_frac"] = batched_metrics[
                        "train/batched/success_frac"
                    ]
            self.logger.record(step_counter - 1, batched_metrics)
            self._flush_logger()
            if progress_bar is not None:
                progress_bar.update({**rollout_step_info, "phase": "rollout"}, n=processed)

        return metrics, rollout_progress, step_counter

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
        self._flush_logger()
        close = getattr(self.logger, "close", None)
        if callable(close):
            close()

    def _flush_logger(self) -> None:
        flush = getattr(self.logger, "flush", None)
        if callable(flush):
            flush()

    def _maybe_train(
        self,
        step: int,
        metrics: dict[str, Any],
        *,
        progress_bar: TqdmProgressBar | None = None,
    ) -> dict[str, Any]:
        if not self._should_train(step):
            return metrics
        if progress_bar is not None:
            progress_bar.pulse({"phase": "training"})
            setattr(
                self.learner,
                "_on_reanalyze_progress",
                lambda done, total: progress_bar.pulse(
                    {
                        "phase": "reanalyze",
                        "reanalyze": f"{done}/{total}",
                    }
                ),
            )
        try:
            trained_metrics = self.learner.train_step(self.replay_buffer)
        finally:
            if progress_bar is not None and hasattr(self.learner, "_on_reanalyze_progress"):
                delattr(self.learner, "_on_reanalyze_progress")
        merged = {**metrics, **trained_metrics}
        if progress_bar is not None:
            progress_bar.pulse({**trained_metrics, "phase": "buffer"})
        return merged

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
