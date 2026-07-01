"""Agent integration tests."""

import gymnasium as gym
import pytest

import algorl as arl
from algorl.agents.configs import EfficientZeroConfig


def test_public_exports() -> None:
    assert hasattr(arl, "EfficientZero")
    assert hasattr(arl, "DreamerV3")
    assert arl.__version__ == "0.0.1"


def test_efficient_zero_construction(cartpole_env: gym.Env) -> None:
    config = EfficientZeroConfig(require_implemented=False)
    agent = arl.EfficientZero(cartpole_env, config=config)
    assert agent.backend.name == "jax"
    assert agent.world_model is not None
    assert agent.planner is not None
    assert agent.planner.world_model is agent.world_model
    assert agent.learner is not None
    assert agent.learner.world_model is agent.world_model
    assert agent.replay_buffer is not None
    assert agent.observation_space == cartpole_env.observation_space
    assert agent.action_space == cartpole_env.action_space


def test_efficient_zero_rejects_stubs_by_default(cartpole_env: gym.Env) -> None:
    with pytest.raises(NotImplementedError, match="Stub component"):
        arl.EfficientZero(cartpole_env)


def test_agent_rejects_invalid_env() -> None:
    with pytest.raises(TypeError, match="Expected gymnasium.Env"):
        arl.EfficientZero(object())  # type: ignore[arg-type]
