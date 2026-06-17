"""Shared Gymnasium-based agent base class."""

from __future__ import annotations

import gymnasium as gym

from algorl.core.agent import BaseAgent
from algorl.envs import check_gymnasium_env


class GymnasiumAgent(BaseAgent):
    """Base agent that operates on a Gymnasium environment."""

    def __init__(self, env: gym.Env) -> None:
        self.env = check_gymnasium_env(env)

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space
