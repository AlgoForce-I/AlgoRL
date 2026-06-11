"""Agent integration tests."""

import gymnasium as gym
import pytest

import algorl as arl


def test_public_exports() -> None:
    assert hasattr(arl, "EfficientZero")
    assert hasattr(arl, "DreamerV3")
    assert arl.__version__ == "0.0.1"


def test_efficient_zero_construction(cartpole_env: gym.Env) -> None:
    agent = arl.EfficientZero(cartpole_env, backend="jax")
    assert agent.backend.name == "jax"
    assert agent.world_model is not None
    assert agent.planner is not None
    assert agent.learner is not None
    assert agent.observation_space is cartpole_env.observation_space
    assert agent.action_space is cartpole_env.action_space


def test_agent_rejects_non_gymnasium_env() -> None:
    with pytest.raises(TypeError, match="gymnasium.Env"):
        arl.EfficientZero(object())  # type: ignore[arg-type]
