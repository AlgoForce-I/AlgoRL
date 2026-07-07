"""Shared agent base class."""

from __future__ import annotations

from typing import Any

from algorl.core.agent import BaseAgent
from algorl.envs import resolve_env
from algorl.envs.training_env import TrainingEnv


class Agent(BaseAgent):
    """Base agent that operates on a Gymnasium or JAX-native environment."""

    def __init__(self, env: object) -> None:
        self.env = resolve_env(env)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


# Backward-compatible alias.
GymnasiumAgent = Agent
