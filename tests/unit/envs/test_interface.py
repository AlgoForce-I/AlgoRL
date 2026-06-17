"""Gymnasium environment validation tests."""

import gymnasium as gym
import pytest

from algorl.envs import check_gymnasium_env


def test_check_gymnasium_env_accepts_gymnasium_env() -> None:
    env = gym.make("CartPole-v1")
    assert check_gymnasium_env(env) is env
    env.close()


def test_check_gymnasium_env_rejects_non_gymnasium_object() -> None:
    with pytest.raises(TypeError, match="gymnasium.Env"):
        check_gymnasium_env(object())
