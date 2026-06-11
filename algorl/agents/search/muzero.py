"""MuZero agent."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class MuZero(GymnasiumAgent):
    """MuZero search-based agent."""

    composition_name: ClassVar[str] = "muzero"

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        components = compose_agent(self.composition_name, backend=backend, **kwargs)
        self.backend = components.backend
        self.world_model = components.world_model
        self.planner = components.planner
        self.learner = components.learner

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        # Implement:
        # 1. Create a replay buffer for MCTS targets and environment transitions.
        # 2. Loop for ``total_timesteps`` using ``self.env.reset()`` / ``self.env.step()``.
        # 3. Select actions with ``self.planner.search(obs)``.
        # 4. Store obs, action, reward, MCTS policy, value targets, and done flags.
        # 5. Call ``self.learner.train_step(buffer)`` once enough data is collected.
        # 6. Optionally log metrics and save checkpoints.
        raise NotImplementedError("MuZero training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        return self.planner.search(observation, deterministic=deterministic)
