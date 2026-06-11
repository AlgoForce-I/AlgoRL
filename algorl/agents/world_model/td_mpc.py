"""TD-MPC agent."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from algorl.agents._base import GymnasiumAgent
from algorl.agents._compose import compose_agent
from algorl.core.types import Action, Observation


class TDMPC(GymnasiumAgent):
    """TD-MPC / TD-MPC2 world-model agent."""

    def __init__(self, env: gym.Env, backend: str = "jax", **kwargs: Any) -> None:
        super().__init__(env)
        self.backend, self.world_model, self.planner, self.learner = compose_agent(
            backend=backend,
            world_model_kind="td_mpc",
            planner_kind="mpc",
            learner_kind="td_mpc",
            **kwargs,
        )

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        # Implement:
        # 1. Collect transitions from ``self.env``.
        # 2. Train the latent TD-MPC model in the learner.
        # 3. Use MPC planning via ``self.planner.search(obs)`` for action selection.
        # 4. Call ``self.learner.train_step(buffer)`` on sampled latent transitions.
        raise NotImplementedError("TDMPC training is not implemented yet.")

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        if self.planner is None:
            raise RuntimeError("TDMPC planner is not configured.")
        return self.planner.search(observation, deterministic=deterministic)
