"""EfficientZero agent."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class EfficientZero(GymnasiumAgent):
    """EfficientZero search-based agent."""

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        self.backend, self.world_model, self.planner, self.learner = compose_agent(
            backend=backend,
            world_model_kind="efficient_zero",
            planner_kind="mcts",
            learner_kind="efficient_zero",
            **kwargs,
        )

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        # Implement: same training loop as MuZero, plus EfficientZero-specific pieces:
        # - self-supervised representation loss in the learner
        # - reanalyze old buffer entries with the latest model
        # - model-based value targets / value prefix if following the paper closely
        # Keep env interaction and orchestration here; put loss math in backends/jax/learners/.
        raise NotImplementedError("EfficientZero training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        if self.planner is None:
            raise RuntimeError("EfficientZero planner is not configured.")
        return self.planner.search(observation, deterministic=deterministic)
