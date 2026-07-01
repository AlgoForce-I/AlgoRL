"""Shared environment interaction and training orchestration."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import BaseAgentConfig
from algorl.common.callbacks import Callback, CallbackList
from algorl.common.checkpoints import save_checkpoint
from algorl.common.logger import Logger
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Action, Observation, Transition
from algorl.envs.jax_env import PolicyFn
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
        self._key = jax.random.PRNGKey(config.seed)

    def run(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None = None,
        extra_step_info: dict[str, Any] | None = None,
    ) -> None:
        """Interact with the environment and call ``learner.train_step`` on schedule."""
        if self.env.is_batched:
            self._run_batched(total_timesteps, checkpoint_path=checkpoint_path, extra_step_info=extra_step_info)
            return
        self._run_sequential(total_timesteps, checkpoint_path=checkpoint_path, extra_step_info=extra_step_info)

    def _run_sequential(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None,
        extra_step_info: dict[str, Any] | None,
    ) -> None:
        observation, _ = self.env.reset(seed=self.config.seed)
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
                    info=info,
                )
            )

            if done:
                observation, _ = self.env.reset()
            else:
                observation = next_observation

            metrics = self._maybe_train(step, metrics)
            self._log_step(step, reward=reward, done=done, metrics=metrics, extra_step_info=extra_step_info)
            self._maybe_checkpoint(step, metrics, checkpoint_path)

    def _run_batched(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None,
        extra_step_info: dict[str, Any] | None,
    ) -> None:
        chunk_size = max(1, self.config.jax_rollout_chunk)
        metrics: dict[str, Any] = {}
        steps_collected = 0
        step_counter = 0

        while steps_collected < total_timesteps:
            chunk_steps = min(chunk_size, total_timesteps - steps_collected)
            self._key, rollout_key = jax.random.split(self._key)
            batch = self.env.collect_rollout(self._batched_policy(), chunk_steps, key=rollout_key)
            transitions = batch.to_transitions()
            for transition in transitions:
                self.replay_buffer.add(transition)
                steps_collected += 1
                metrics = self._maybe_train(step_counter, metrics)
                self._log_step(
                    step_counter,
                    reward=transition.reward,
                    done=transition.done,
                    metrics=metrics,
                    extra_step_info=extra_step_info,
                )
                self._maybe_checkpoint(step_counter, metrics, checkpoint_path)
                step_counter += 1
                if steps_collected >= total_timesteps:
                    break

    def _batched_policy(self) -> PolicyFn:
        def policy(observations: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
            del key
            actions = []
            for lane in range(observations.shape[0]):
                action = self._select_action(np.asarray(observations[lane], dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
            return jnp.asarray(actions, dtype=jnp.float32)

        return policy

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
    ) -> None:
        step_info = {"reward": float(reward), "done": done, **metrics}
        if extra_step_info:
            step_info.update(extra_step_info)
        self.logger.record(step, step_info)
        self.callbacks.on_step(step, step_info)

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
        return self.planner.search(observation, deterministic=True)

    def _should_train(self, step: int) -> bool:
        if step < self.config.learning_starts:
            return False
        if self.config.train_freq <= 0 or step % self.config.train_freq != 0:
            return False
        return len(self.replay_buffer) >= self.config.batch_size
