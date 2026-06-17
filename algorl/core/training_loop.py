"""Shared environment interaction and training orchestration."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from algorl.agents.configs import BaseAgentConfig
from algorl.common.callbacks import Callback, CallbackList
from algorl.common.checkpoints import save_checkpoint
from algorl.common.logger import Logger
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Action, Observation, Transition


class TrainingLoop:
    """Collect environment data, train on a schedule, and emit metrics."""

    def __init__(
        self,
        *,
        env: gym.Env,
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

    def run(
        self,
        total_timesteps: int,
        *,
        checkpoint_path: str | None = None,
        extra_step_info: dict[str, Any] | None = None,
    ) -> None:
        """Interact with the environment and call ``learner.train_step`` on schedule."""
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

            if self._should_train(step):
                metrics = self.learner.train_step(self.replay_buffer)

            step_info = {"reward": float(reward), "done": done, **metrics}
            if extra_step_info:
                step_info.update(extra_step_info)

            self.logger.record(step, step_info)
            self.callbacks.on_step(step, step_info)

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
