"""MuZero agent."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class MuZero(GymnasiumAgent):
    """MuZero search-based agent."""

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        self.backend, self.world_model, self.planner, self.learner = compose_agent(
            backend=backend,
            world_model_kind="muzero",
            planner_kind="mcts",
            learner_kind="muzero",
            **kwargs,
        )

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        raise NotImplementedError("MuZero training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        if self.planner is None:
            raise RuntimeError("MuZero planner is not configured.")
        return self.planner.search(observation, deterministic=deterministic)
