"""Environment wrappers."""

from __future__ import annotations

from typing import Any

from algorl.envs.interface import EnvAdapter, EnvLike


class ActionRepeatWrapper(EnvAdapter):
    """Repeat the same action for multiple environment steps."""

    def __init__(self, env: EnvLike, repeat: int = 1) -> None:
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = repeat

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        observation: Any = None
        info: dict[str, Any] = {}

        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info
