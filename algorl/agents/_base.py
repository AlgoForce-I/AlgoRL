"""Shared agent base class."""

from __future__ import annotations

from algorl.agents.configs import BaseAgentConfig
from algorl.core.agent import BaseAgent
from algorl.envs.resolve import make_env
from algorl.envs.training_env import TrainingEnv


class Agent(BaseAgent):
    """Base agent that operates on a Gymnasium or JAX-native environment."""

    def __init__(self, env: object, *, config: BaseAgentConfig | None = None) -> None:
        seed = config.seed if config is not None else 0
        self.env: TrainingEnv = make_env(env, config=config, seed=seed)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


# Backward-compatible alias.
GymnasiumAgent = Agent
