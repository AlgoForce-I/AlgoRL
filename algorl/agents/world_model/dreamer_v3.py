"""DreamerV3 agent."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class DreamerV3(GymnasiumAgent):
    """DreamerV3 world-model agent."""

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        self.backend, self.world_model, self.planner, self.learner = compose_agent(
            backend=backend,
            world_model_kind="rssm",
            planner_kind="imagination",
            learner_kind="dreamer",
            **kwargs,
        )

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        raise NotImplementedError("DreamerV3 training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        if self.planner is None:
            raise RuntimeError("DreamerV3 planner is not configured.")
        return self.planner.search(observation, deterministic=deterministic)
