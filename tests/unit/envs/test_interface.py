"""Gymnasium and JAX environment validation tests."""

import gymnasium as gym
import pytest

from algorl.envs import check_env, resolve_env


def test_check_env_accepts_gymnasium_env() -> None:
    env = gym.make("CartPole-v1")
    training_env = check_env(env)
    assert training_env.observation_space == env.observation_space
    env.close()


def test_resolve_env_wraps_gymnasium_env() -> None:
    env = gym.make("CartPole-v1")
    training_env = resolve_env(env)
    assert not training_env.is_batched
    env.close()


def test_check_env_rejects_invalid_object() -> None:
    with pytest.raises(TypeError, match="Expected gymnasium.Env"):
        check_env(object())
