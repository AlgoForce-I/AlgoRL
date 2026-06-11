"""Gymnasium environment wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat the same action for multiple environment steps."""

    def __init__(self, env: gym.Env, repeat: int = 1) -> None:
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = repeat

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        observation: ObsType = None  # type: ignore[assignment]
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info
