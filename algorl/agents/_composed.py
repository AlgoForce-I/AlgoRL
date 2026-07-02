"""Shared composed-agent base class."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym

from algorl.agents._base import Agent
from algorl.agents._compose import compose_agent
from algorl.agents.configs import BaseAgentConfig
from algorl.core.training_loop import TrainingLoop
from algorl.core.types import Action, Observation
from algorl.envs.training_env import TrainingEnv


class ComposedAgent(Agent):
    """Agent that wires world model, planner, learner, and buffer from a composition."""

    composition_name: ClassVar[str]
    config_class: ClassVar[type[BaseAgentConfig]]

    def __init__(self, env: TrainingEnv | object, config: BaseAgentConfig | None = None) -> None:
        super().__init__(env)
        self.config = config or self.config_class()
        components = compose_agent(
            self.composition_name,
            env=self.env,
            config=self.config,
        )
        self.backend = components.backend
        self.world_model = components.world_model
        self.planner = components.planner
        self.learner = components.learner
        self.replay_buffer = components.replay_buffer

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        callbacks = kwargs.pop("callbacks", None)
        logger = kwargs.pop("logger", None)
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        tensorboard = bool(kwargs.pop("tensorboard", False))
        tensorboard_log_dir = kwargs.pop("tensorboard_log_dir", None)
        progress_bar = kwargs.pop("progress_bar", None)
        progress_bar_kwargs = kwargs.pop("progress_bar_kwargs", None)

        if logger is None and (tensorboard or tensorboard_log_dir is not None):
            log_dir = tensorboard_log_dir or f"runs/{self.composition_name}"
            from algorl.common.tensorboard_logger import TensorboardLogger

            logger = TensorboardLogger(log_dir)

        if progress_bar is True:
            from algorl.common.progress_bar import TqdmProgressBar

            bar_kwargs = progress_bar_kwargs or {}
            progress_bar = TqdmProgressBar(**bar_kwargs)
        elif progress_bar is False:
            progress_bar = None

        loop = TrainingLoop(
            env=self.env,
            planner=self.planner,
            learner=self.learner,
            replay_buffer=self.replay_buffer,
            config=self.config,
            callbacks=callbacks,
            logger=logger,
        )
        loop.run(
            total_timesteps,
            checkpoint_path=checkpoint_path,
            extra_step_info=kwargs or None,
            progress_bar=progress_bar,
        )

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        return self.planner.search(observation, deterministic=deterministic)
