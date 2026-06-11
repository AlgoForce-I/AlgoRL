"""PlaNet agent."""

from __future__ import annotations

from typing import Any

from algorl.agents._compose import compose_agent
from algorl.core.agent import BaseAgent
from algorl.core.types import Action, Observation


class PlaNet(BaseAgent):
    """PlaNet world-model agent."""

    def __init__(self, env: Any, backend: str = "jax", **kwargs: Any) -> None:
        self.env = env
        self.backend, self.world_model, self.planner, self.learner = compose_agent(
            backend=backend,
            world_model_kind="rssm",
            planner_kind="cem",
            learner_kind="planet",
            **kwargs,
        )

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        raise NotImplementedError("PlaNet training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        if self.planner is None:
            raise RuntimeError("PlaNet planner is not configured.")
        return self.planner.search(observation, deterministic=deterministic)
