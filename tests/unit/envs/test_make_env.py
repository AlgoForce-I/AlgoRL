"""Training environment factory tests."""

import gymnasium as gym
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.envs import make_env


def test_make_env_vectorizes_gym_env_for_batched_config() -> None:
    config = EfficientZeroConfig.for_batched(num_envs=4)
    source = gym.make("CartPole-v1")
    env = make_env(source, config=config, seed=0)
    assert env.is_batched
    assert env.num_envs == 4
    env.raw.vector_env.close()
    source.close()


def test_make_env_vectorizes_gym_id_for_batched_config() -> None:
    config = EfficientZeroConfig.for_batched(num_envs=4)
    env = make_env("CartPole-v1", config=config, seed=0)
    assert env.is_batched
    assert env.num_envs == 4
    env.raw.vector_env.close()


def test_make_env_wraps_sequential_gym_env_for_search() -> None:
    config = EfficientZeroConfig.for_sequential()
    source = gym.make("CartPole-v1")
    env = make_env(source, config=config, seed=0)
    assert not env.is_batched
    assert env.num_envs == 1
    assert hasattr(env.raw, "search_environment")
    env.raw.env.close()
    source.close()


def test_make_env_validates_rollout_lane_mismatch() -> None:
    config = EfficientZeroConfig.for_batched(num_envs=8)
    env = make_env("CartPole-v1", seed=0, num_envs=4)
    with pytest.raises(ValueError, match="expects 8 parallel rollout env"):
        make_env(env, config=config)
    env.raw.vector_env.close()

