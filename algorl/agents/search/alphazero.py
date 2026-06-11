"""AlphaZero agent."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class AlphaZero(GymnasiumAgent):
    """AlphaZero search-based agent."""

    composition_name: ClassVar[str] = "alphazero"

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        components = compose_agent(self.composition_name, backend=backend, **kwargs)
        self.backend = components.backend
        self.world_model = components.world_model
        self.planner = components.planner
        self.learner = components.learner

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        # Implement:
        # 1. Self-play loop on ``self.env`` using MCTS with the current policy/value network.
        # 2. Store MCTS visit counts as policy targets and game outcomes as value targets.
        # 3. Train with ``self.learner.train_step(buffer)``; no learned dynamics are required.
        raise NotImplementedError("AlphaZero training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        return self.planner.search(observation, deterministic=deterministic)
