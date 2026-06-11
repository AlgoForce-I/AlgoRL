"""PlaNet agent."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class PlaNet(GymnasiumAgent):
    """PlaNet world-model agent."""

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        self.backend, self.world_model, self.planner, self.learner = compose_agent(
            backend=backend,
            world_model_kind="rssm",
            planner_kind="cem",
            learner_kind="planet",
            **kwargs,
        )

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
        if self.planner is None:
            raise RuntimeError("PlaNet planner is not configured.")
        return self.planner.search(observation, deterministic=deterministic)
