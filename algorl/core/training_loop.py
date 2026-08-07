"""Shared environment interaction and training orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import BaseAgentConfig
from algorl.common.callbacks import Callback, CallbackList
from algorl.common.checkpoints import (
    load_run_checkpoint,
    prune_step_checkpoints,
    save_run_checkpoint,
    write_json,
)
from algorl.common.episode_metrics import (
    BatchedEpisodeMetricsTracker,
    EpisodeMetricsTracker,
    batched_episode_summary_metrics,
)
from algorl.backends.jax.memory import collect_memory_metrics
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
        agent_name: str = "agent",
    ) -> None:
        self.env = env
        self.planner = planner
        self.learner = learner
        self.replay_buffer = replay_buffer
        self.config = config
        self.agent_name = agent_name
        self.callbacks = callbacks if isinstance(callbacks, CallbackList) else CallbackList(
            [callbacks] if callbacks is not None else []
        )
        self.logger = logger or Logger()
        if env.is_batched:
            self._episode_tracker: EpisodeMetricsTracker | BatchedEpisodeMetricsTracker = (
                BatchedEpisodeMetricsTracker(env.num_envs)
            )
        else:
            self._episode_tracker = EpisodeMetricsTracker()
        self._key = jax.random.PRNGKey(config.seed)
        self._progress_bar: TqdmProgressBar | None = None
        # First global step at which training may run; pushed forward at
        # continual-learning task switches to replay the learning_starts warmup.
        self._min_train_step = 0
        self._last_memory_log_step = -1
        self._checkpoint_dir: str | None = getattr(config, "checkpoint_dir", None)
        self._recent_returns: list[float] = []
        self._best_score = float("-inf")
        self._skip_initial_env_reset = False

    def restore_run_checkpoint(self, directory: str | Path) -> dict[str, Any]:
        """Load learner/buffer/env/logger state from a multi-file checkpoint."""
        payload = load_run_checkpoint(
            directory=directory,
            learner=self.learner,
            replay_buffer=self.replay_buffer,
            env=self.env,
            rng_key=self._key,
        )
        loop = payload["loop"]
        self._min_train_step = int(loop.get("min_train_step", 0))
        self._last_memory_log_step = int(loop.get("last_memory_log_step", -1))
        self._best_score = float(loop.get("best_score", float("-inf")))
        self._recent_returns = [float(v) for v in loop.get("recent_returns", [])]
        key_data = loop.get("rng_key")
        if key_data is not None:
            self._key = jnp.asarray(key_data, dtype=jnp.uint32)
        logger_payload = payload.get("logger")
        if logger_payload is not None and "history" in logger_payload:
            self.logger.history = list(logger_payload["history"])
        self._skip_initial_env_reset = True
        return payload

    def run(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None = None,
        checkpoint_dir: str | None = None,
        start_step: int = 0,
        extra_step_info: dict[str, Any] | None = None,
        progress_bar: TqdmProgressBar | None = None,
    ) -> None:
        """Interact with the environment and call ``learner.train_step`` on schedule."""
        if checkpoint_dir is not None:
            self._checkpoint_dir = checkpoint_dir
        elif checkpoint_path is not None and self._checkpoint_dir is None:
            # Treat legacy path as a checkpoint directory root.
            self._checkpoint_dir = checkpoint_path
        self._progress_bar = progress_bar
        if progress_bar is not None:
            progress_bar.start(total_timesteps, initial=max(0, int(start_step)))
        try:
            if self.env.is_batched:
                self._run_batched(
                    total_timesteps,
                    start_step=start_step,
                    extra_step_info=extra_step_info,
                    progress_bar=progress_bar,
                )
                return
            self._run_sequential(
                total_timesteps,
                start_step=start_step,
                extra_step_info=extra_step_info,
                progress_bar=progress_bar,
            )
        finally:
            self._progress_bar = None
            if progress_bar is not None:
                progress_bar.close()
            self._close_logger()

    def _run_sequential(
        self,
        total_timesteps: int,
        *,
        start_step: int,
        extra_step_info: dict[str, Any] | None,
        progress_bar: TqdmProgressBar | None,
    ) -> None:
        if self._skip_initial_env_reset and start_step > 0:
            observation, reset_info = self.env.reset(seed=None)
        else:
            observation, reset_info = self.env.reset(seed=self.config.seed)
        self._episode_tracker.begin_episode(reset_info)
        metrics: dict[str, Any] = {}
        self._sync_rollout_task_id()

        for step in range(start_step, total_timesteps):
            self._sync_rollout_task_id()
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

            enriched_info = self._enrich_transition_info(dict(info))
            episode_event = self._episode_tracker.observe_step(float(reward), done, enriched_info)
            if episode_event is not None:
                metrics = self._record_episode(step, episode_event, metrics)

            if info.get("task_changed"):
                self._notify_task_boundary(info)
                clear_buffer = getattr(self.replay_buffer, "clear", None)
                if callable(clear_buffer):
                    clear_buffer()
                self._min_train_step = step + 1 + self.config.learning_starts
                self._reset_autosave_best_for_new_task()
                self._maybe_boundary_checkpoint(step, metrics)

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
            self._maybe_checkpoint(step, metrics)
            if progress_bar is not None:
                progress_bar.update(step_info)

    def _run_batched(
        self,
        total_timesteps: int,
        *,
        start_step: int,
        extra_step_info: dict[str, Any] | None,
        progress_bar: TqdmProgressBar | None,
    ) -> None:
        chunk_size = max(1, self.config.jax_rollout_chunk)
        metrics: dict[str, Any] = {}
        steps_collected = max(0, int(start_step))
        step_counter = max(0, int(start_step))

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

            if getattr(self.config, "sync_self_play_before_rollout", False):
                sync_self_play = getattr(self.learner, "sync_self_play_for_rollout", None)
                if callable(sync_self_play):
                    sync_self_play()

            self._sync_rollout_task_id()

            batch = self.env.collect_rollout(
                self._batched_policy(search_results),
                chunk_steps,
                key=rollout_key,
                on_step=on_rollout_step,
            )
            transitions = self._transitions_from_rollout(batch, search_results)
            self._release_rollout_search_cache(search_results)
            transitions, dropped_steps = self._apply_task_boundary(
                transitions,
                chunk_start_step,
            )
            steps_collected += dropped_steps
            if self.config.gradient_steps_per_rollout is None:
                # Sequential-parity: add transitions and train immediately, so the
                # replay-buffer warmup within the chunk matches the sequential
                # loop's timing (affects PER sampling and learning curve shape).
                num_added = 0
                for transition in transitions:
                    if steps_collected >= total_timesteps:
                        break
                    self.replay_buffer.add(transition)
                    steps_collected += 1
                    num_added += 1
                    current_step = chunk_start_step + dropped_steps + num_added - 1
                    metrics = self._maybe_train(
                        current_step,
                        metrics,
                        progress_bar=progress_bar,
                    )
                    self._maybe_checkpoint(current_step, metrics)
                if num_added == 0:
                    continue
            else:
                num_added = 0
                for transition in transitions:
                    self.replay_buffer.add(transition)
                    steps_collected += 1
                    num_added += 1
                    if steps_collected >= total_timesteps:
                        break

                if num_added == 0:
                    continue
                first_added_step = chunk_start_step + dropped_steps
                chunk_end_step = first_added_step + num_added - 1
                train_steps = self._batched_gradient_steps(first_added_step, chunk_end_step)
                metrics = self._run_gradient_burst(
                    train_steps,
                    metrics,
                    progress_bar=progress_bar,
                )

    def _apply_task_boundary(
        self,
        transitions: list[Transition],
        chunk_start_step: int,
    ) -> tuple[list[Transition], int]:
        """Start a continual-learning task with fresh plasticity.

        Envs mark the last transitions of a finished task with
        ``info["task_changed"]``. When that flag is seen, the replay buffer is
        flushed (old-task data must not train the new task), the learner resets
        its optimizer/schedule state, and training pauses for a fresh
        ``learning_starts`` warmup. Transitions up to and including the boundary
        are dropped (they would be flushed with the buffer anyway); their count
        is returned so the caller keeps global step accounting intact.
        """
        boundary = None
        for index, transition in enumerate(transitions):
            if transition.info.get("task_changed"):
                boundary = index
        if boundary is None:
            return transitions, 0

        self._notify_task_boundary(transitions[boundary].info)

        clear_buffer = getattr(self.replay_buffer, "clear", None)
        if callable(clear_buffer):
            clear_buffer()
        boundary_step = chunk_start_step + boundary + 1
        self._min_train_step = boundary_step + self.config.learning_starts
        self._reset_autosave_best_for_new_task()
        self._maybe_boundary_checkpoint(boundary_step, {})
        return transitions[boundary + 1:], boundary + 1

    def _notify_task_boundary(self, info: dict[str, Any]) -> None:
        """Invoke learner continual-learning hook when the env switches tasks."""
        finished_task = info.get("seq_idx", info.get("task_index", 0))
        new_task_id = int(finished_task) + 1
        on_task_boundary = getattr(self.learner, "on_task_boundary", None)
        if callable(on_task_boundary):
            on_task_boundary(new_task_id)

    def _sync_rollout_task_id(self) -> None:
        """Keep learner materialization aligned with the env's active task."""
        task_id = self._env_current_task_index()
        if task_id is None:
            return
        current = int(getattr(self.learner, "task_id", -1))
        if current == task_id:
            return
        if task_id > current:
            on_task_boundary = getattr(self.learner, "on_task_boundary", None)
            if callable(on_task_boundary):
                on_task_boundary(task_id)
                return
        set_task_id = getattr(self.learner, "set_task_id", None)
        if callable(set_task_id):
            set_task_id(task_id)

    def _env_current_task_index(self) -> int | None:
        for candidate in (
            self.env,
            getattr(self.env, "raw", None),
            getattr(self.env, "unwrapped", None),
        ):
            if candidate is None:
                continue
            current = getattr(candidate, "current_task_index", None)
            if current is None:
                continue
            return int(current() if callable(current) else current)
        return None

    def _batched_gradient_steps(self, chunk_start: int, chunk_end: int) -> list[int]:
        """Return global env steps that should trigger ``train_step`` after one rollout chunk."""
        capped = self.config.gradient_steps_per_rollout
        if capped is None:
            return [
                step
                for step in range(chunk_start, chunk_end + 1)
                if self._should_train(step)
            ]
        if chunk_end < max(self.config.learning_starts, self._min_train_step):
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

    def _run_gradient_burst(
        self,
        train_steps: list[int],
        metrics: dict[str, Any],
        *,
        progress_bar: TqdmProgressBar | None,
    ) -> dict[str, Any]:
        """Run a capped post-rollout training burst (batched throughput path)."""
        if not train_steps:
            return metrics

        num_steps = len(train_steps)
        if progress_bar is not None:
            progress_bar.pulse({"phase": "training", "train_burst": f"0/{num_steps}"})
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
            train_burst = getattr(self.learner, "train_burst", None)
            if callable(train_burst):
                trained_metrics = train_burst(
                    self.replay_buffer,
                    num_steps,
                    on_progress=lambda done, total: (
                        progress_bar.pulse(
                            {
                                "phase": "training",
                                "train_burst": f"{done}/{total}",
                            }
                        )
                        if progress_bar is not None
                        else None
                    ),
                )
            else:
                trained_metrics: dict[str, float] = {}
                for index, train_step in enumerate(train_steps):
                    if progress_bar is not None:
                        progress_bar.pulse(
                            {
                                "phase": "training",
                                "train_burst": f"{index + 1}/{num_steps}",
                            }
                        )
                    trained_metrics = self.learner.train_step(self.replay_buffer)
                    self._maybe_checkpoint(train_step, {**metrics, **trained_metrics})
        finally:
            if progress_bar is not None and hasattr(self.learner, "_on_reanalyze_progress"):
                delattr(self.learner, "_on_reanalyze_progress")
            self._release_rollout_search_cache([])

        merged = {**metrics, **trained_metrics}
        # Post-train sample: captures VRAM after the heavy AD / rematerialize peak.
        self._maybe_record_memory(train_steps[-1], force=True)
        self._maybe_checkpoint(train_steps[-1], merged)
        if progress_bar is not None:
            progress_bar.pulse({**trained_metrics, "phase": "buffer"})
        return merged

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
        completed_episodes: list = []

        for lane in range(min(int(env_steps), reward_array.shape[0], len(lane_infos))):
            if steps_collected + rollout_progress >= total_timesteps:
                break
            info = dict(lane_infos[lane])
            search_info = self._search_info_from_result(
                getattr(self.planner, "last_result", None),
                lane,
            )
            if search_info:
                info.update(search_info)
            reward = float(reward_array[lane])
            done = bool(done_array[lane])
            if "success" in info:
                success_values.append(float(info["success"]))

            if isinstance(self._episode_tracker, BatchedEpisodeMetricsTracker):
                episode_event = self._episode_tracker.observe_step(lane, reward, done, info)
            else:
                episode_event = self._episode_tracker.observe_step(reward, done, info)
            if episode_event is not None:
                metrics = self._record_episode(step_counter, episode_event, metrics)
                completed_episodes.append(episode_event)

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
            if completed_episodes:
                batched_metrics.update(batched_episode_summary_metrics(completed_episodes))
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
        progress_bar = self._progress_bar

        if isinstance(planner, BatchedPlanner):
            def policy(observations: jnp.ndarray | np.ndarray, key: jnp.ndarray) -> jnp.ndarray:
                del key
                if progress_bar is not None:
                    progress_bar.pulse({"phase": "search"})
                obs_array = np.asarray(observations, dtype=np.float32)
                obs_batch = [obs_array[lane] for lane in range(obs_array.shape[0])]
                result = planner.search_batch(obs_batch, deterministic=False)
                snapshot_fn = getattr(result, "as_training_snapshot", None)
                snapshot = snapshot_fn() if callable(snapshot_fn) else result
                self.planner.last_result = snapshot
                if search_results is not None:
                    search_results.append(snapshot)
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
                step_info = (
                    batch.step_info[step_idx][env_idx]
                    if batch.step_info is not None
                    else None
                )
                # Gymnasium NEXT_STEP autoreset: the step after a done lane
                # ignores the action and returns the reset observation; it is
                # not a real transition and must not enter the replay buffer.
                if step_info is not None and step_info.get("replay_skip"):
                    continue
                info = self._search_info_from_result(result, env_idx)
                if step_info is not None:
                    info = {**step_info, **info}
                # Lane identity lets trajectory-based buffers keep each env's
                # stream temporally coherent despite interleaved adds.
                info["env_id"] = env_idx
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
        from algorl.common.episode_metrics import EpisodeEndEvent, episode_metrics_from_event

        if not isinstance(event, EpisodeEndEvent):
            return metrics
        if isinstance(self.logger, TensorboardLogger):
            episode_metrics = self.logger.record_episode(step, event)
        else:
            episode_metrics = episode_metrics_from_event(event)
        merged = {**metrics, **episode_metrics}
        self._maybe_autosave_best(step, float(event.episode_return), merged)
        return merged

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
            self._release_rollout_search_cache([])
        merged = {**metrics, **trained_metrics}
        self._maybe_record_memory(step, force=True)
        if progress_bar is not None:
            progress_bar.pulse({**trained_metrics, "phase": "buffer"})
        return merged

    def _release_rollout_search_cache(self, search_results: list[Any]) -> None:
        """Drop rollout MCTS payloads before learner reanalyze allocates GPU memory."""
        search_results.clear()
        if hasattr(self.planner, "last_result"):
            self.planner.last_result = None

    def _maybe_record_memory(self, step: int, *, force: bool = False) -> None:
        """Write RAM/VRAM scalars to the logger (TensorBoard when enabled).

        ``memory_log_interval <= 0`` disables all memory logging.
        ``force=True`` logs even if the interval has not elapsed (used after
        gradient bursts to capture post-train VRAM peaks).
        """
        interval = int(getattr(self.config, "memory_log_interval", 1_000))
        if interval <= 0:
            return
        if (
            not force
            and self._last_memory_log_step >= 0
            and step - self._last_memory_log_step < interval
        ):
            return
        metrics = collect_memory_metrics()
        if not metrics:
            return
        self.logger.record(int(step), metrics)
        if isinstance(self.logger, TensorboardLogger):
            self.logger.flush()
        self._last_memory_log_step = int(step)

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
        # Callbacks may enrich ``step_info`` before it is recorded (e.g. CL retention).
        self.callbacks.on_step(step, step_info)
        self.logger.record(step, step_info)
        self._maybe_record_memory(step)
        return step_info

    def _loop_checkpoint_state(self, step: int) -> dict[str, Any]:
        task_id = getattr(self.learner, "task_id", self._env_current_task_index())
        return {
            "step": int(step),
            "min_train_step": int(self._min_train_step),
            "last_memory_log_step": int(self._last_memory_log_step),
            "task_id": None if task_id is None else int(task_id),
            "best_score": float(self._best_score),
            "recent_returns": list(self._recent_returns),
            "rng_key": np.asarray(self._key).tolist(),
        }

    def _write_run_checkpoint(
        self,
        step: int,
        *,
        tag: str | None = None,
        subdirectory: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self._checkpoint_dir:
            return None
        root = Path(self._checkpoint_dir)
        target = root / subdirectory if subdirectory else root / f"step_{int(step):09d}"
        return save_run_checkpoint(
            directory=target,
            agent_name=self.agent_name,
            step=step,
            config=self.config,
            learner=self.learner,
            replay_buffer=self.replay_buffer,
            env=self.env,
            loop_state=self._loop_checkpoint_state(step),
            logger_history=list(getattr(self.logger, "history", [])),
            tag=tag,
            extra_meta=extra_meta,
        )

    def _maybe_checkpoint(self, step: int, metrics: dict[str, Any]) -> None:
        del metrics
        freq = getattr(self.config, "checkpoint_freq", None)
        if (
            self._checkpoint_dir is not None
            and freq is not None
            and step > 0
            and step % int(freq) == 0
        ):
            self._write_run_checkpoint(step, tag="periodic")
            keep = getattr(self.config, "checkpoint_keep_last", None)
            if keep is not None:
                prune_step_checkpoints(self._checkpoint_dir, keep_last=int(keep))

    def _maybe_boundary_checkpoint(self, step: int, metrics: dict[str, Any]) -> None:
        del metrics
        if not self._checkpoint_dir:
            return
        if not getattr(self.config, "checkpoint_at_task_boundary", True):
            return
        finished = int(getattr(self.learner, "task_id", 1)) - 1
        if finished < 0:
            finished = 0
        self._write_run_checkpoint(
            step,
            tag=f"boundary_task_{finished}",
            subdirectory=f"boundary_task_{finished}",
            extra_meta={"buffer_cleared": True, "finished_task": finished},
        )

    def _reset_autosave_best_for_new_task(self) -> None:
        """Clear the global best bar so each CL task can claim its own best."""
        if not getattr(self.config, "autosave_best_per_task", False):
            return
        self._best_score = float("-inf")
        self._recent_returns = []

    def _autosave_best_target(self) -> tuple[str, str]:
        """Return ``(tag, subdirectory)`` for the next autosave-best write."""
        if getattr(self.config, "autosave_best_per_task", False):
            task_id = getattr(self.learner, "task_id", self._env_current_task_index())
            if task_id is not None:
                name = f"best_task_{int(task_id)}"
                return name, name
        return "best", "best"

    def _maybe_autosave_best(
        self,
        step: int,
        episode_return: float,
        metrics: dict[str, Any],
    ) -> None:
        if not getattr(self.config, "autosave_best", False):
            return
        if not self._checkpoint_dir:
            return
        min_step = int(getattr(self.config, "autosave_best_min_step", 0))
        if step < min_step:
            return
        window = max(1, int(getattr(self.config, "autosave_best_window", 10)))
        self._recent_returns.append(float(episode_return))
        if len(self._recent_returns) > window:
            self._recent_returns = self._recent_returns[-window:]
        metric_name = str(getattr(self.config, "autosave_best_metric", "mean_episode_return"))
        if metric_name == "mean_episode_return":
            score = float(np.mean(self._recent_returns))
        elif metric_name in metrics:
            score = float(metrics[metric_name])
        else:
            score = float(episode_return)
        if score <= self._best_score:
            return
        self._best_score = score
        tag, subdirectory = self._autosave_best_target()
        self._write_run_checkpoint(
            step,
            tag=tag,
            subdirectory=subdirectory,
            extra_meta={
                "best_score": score,
                "best_metric": metric_name,
                "best_window": window,
            },
        )
        score_name = (
            f"{subdirectory}_score.json"
            if subdirectory != "best"
            else "best_score.json"
        )
        write_json(
            Path(self._checkpoint_dir) / score_name,
            {
                "best_score": score,
                "best_step": int(step),
                "metric": metric_name,
                "window": window,
                "subdirectory": subdirectory,
            },
        )

    def _select_action(self, observation: Observation) -> Action:
        if self._progress_bar is not None:
            self._progress_bar.pulse({"phase": "search"})
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
        if step < max(self.config.learning_starts, self._min_train_step):
            return False
        if self.config.train_freq <= 0 or step % self.config.train_freq != 0:
            return False
        return len(self.replay_buffer) >= self.config.batch_size
