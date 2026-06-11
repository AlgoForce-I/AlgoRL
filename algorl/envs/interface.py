"""Gymnasium environment utilities."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import Env

__all__ = ["Env", "check_gymnasium_env", "gym"]


def check_gymnasium_env(env: object) -> Env:
    """Validate that ``env`` is a Gymnasium environment."""
    if not isinstance(env, gym.Env):
        raise TypeError(
            f"Expected a gymnasium.Env instance, got {type(env)!r}. "
            "Create environments with gymnasium.make() or subclass gymnasium.Env."
        )
    return env
