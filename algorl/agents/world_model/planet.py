"""PlaNet agent."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class PlaNet(GymnasiumAgent):
    """PlaNet world-model agent."""

    composition_name: ClassVar[str] = "planet"

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        components = compose_agent(self.composition_name, backend=backend, **kwargs)
        self.backend = components.backend
        self.world_model = components.world_model
        self.planner = components.planner
        self.learner = components.learner

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        # Implement:
        # 1. Collect trajectories from ``self.env``.
        # 2. Train the RSSM with an ELBO-style objective in the learner.
        # 3. Use CEM planning via ``self.planner.search(obs)`` for control.
        # 4. Call ``self.learner.train_step(buffer)`` on sequence batches.
        raise NotImplementedError("PlaNet training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        return self.planner.search(observation, deterministic=deterministic)
