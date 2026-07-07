"""Gymnasium and JAX environment utilities."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import Env

from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv
from algorl.envs.resolve import resolve_env
from algorl.envs.training_env import TrainingEnv

__all__ = ["Env", "JaxEnv", "BatchedJaxEnv", "TrainingEnv", "check_env", "gym", "resolve_env"]


def check_env(env: object) -> TrainingEnv:
    """Validate and normalize an environment for AlgoRL agents."""
    return resolve_env(env)
