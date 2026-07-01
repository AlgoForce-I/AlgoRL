"""Gymnasium facade over a :class:`~algorl.envs.jax_env.JaxEnv` or batched adapter."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv
from algorl.envs.training_env import TrainingEnv


class JaxGymEnv(gym.Env):
    """Thin Gymnasium wrapper that delegates to :class:`~algorl.envs.training_env.TrainingEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, jax_env: JaxEnv | BatchedJaxEnv, *, seed: int = 0) -> None:
        super().__init__()
        self._training = TrainingEnv.from_jax(jax_env, seed=seed)
        self.observation_space = self._training.observation_space
        self.action_space = self._training.action_space

    @property
    def jax_env(self) -> JaxEnv | BatchedJaxEnv:
        return self._training.raw

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        return self._training.reset(seed=seed)

    def step(self, action: np.ndarray | list[float]):
        return self._training.step(action)

    def search_environment(self) -> Any:
        search_env = self._training.search_environment()
        if search_env is None:
            raise RuntimeError(f"{type(self).__name__} has no search environment.")
        return search_env
