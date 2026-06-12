"""Shared composed-agent base class."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.agents.configs import BaseAgentConfig
from algorl.core.training_loop import TrainingLoop
from algorl.core.types import Action, Observation


class ComposedAgent(GymnasiumAgent):
    """Agent that wires world model, planner, learner, and buffer from a composition."""

    composition_name: ClassVar[str]
    config_class: ClassVar[type[BaseAgentConfig]]

    def __init__(self, env: gym.Env, config: BaseAgentConfig | None = None) -> None:
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
        )

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        return self.planner.search(observation, deterministic=deterministic)
