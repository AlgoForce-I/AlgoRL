"""EfficientZero world model tests."""

from __future__ import annotations

import gymnasium as gym
import jax.numpy as jnp
import pytest

from algorl.agents._compose import compose_agent
from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.factory import JAXComponentFactory
from algorl.backends.jax.world_models.efficientzero import (
    EfficientZeroLatentState,
    EfficientZeroWorldModel,
    build_efficient_zero_world_model,
)
from algorl.core.component_context import ComponentContext
from algorl.core.factory import get_backend
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_training_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


@pytest.fixture
def world_model(cartpole_training_env: TrainingEnv) -> EfficientZeroWorldModel:
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    return build_efficient_zero_world_model(context)


def test_world_model_registered_as_implemented() -> None:
    factory = JAXComponentFactory()
    assert not factory.is_stub("world_model", "efficient_zero")


def test_compose_agent_uses_efficient_zero_world_model(cartpole_training_env: TrainingEnv) -> None:
    config = EfficientZeroConfig(require_implemented=False)
    components = compose_agent("efficient_zero", env=cartpole_training_env, config=config)
    assert isinstance(components.world_model, EfficientZeroWorldModel)


def test_encode_returns_latent_state(world_model: EfficientZeroWorldModel) -> None:
    latent = world_model.encode(jnp.zeros((4,), dtype=jnp.float32))
    assert isinstance(latent, EfficientZeroLatentState)
    assert latent.state.shape == (world_model.config.hidden_shape,)


def test_transition_reward_and_value(world_model: EfficientZeroWorldModel) -> None:
    latent = world_model.encode(jnp.zeros((4,), dtype=jnp.float32))
    next_latent = world_model.transition(latent, 0)
    assert next_latent.state.shape == latent.state.shape

    reward = world_model.reward(latent, 0)
    assert jnp.asarray(reward).shape == ()

    value = world_model.value(latent)
    assert jnp.asarray(value).shape == ()


def test_initial_and_recurrent_steps(world_model: EfficientZeroWorldModel) -> None:
    obs = jnp.zeros((4,), dtype=jnp.float32)
    latent, value, policy = world_model.initial_step(obs, training=True)
    assert latent.state.shape == (world_model.config.hidden_shape,)
    assert value.shape == (world_model.config.v_num, world_model.config.support_bins)
    assert policy.shape == (world_model.env.action_dim * 2,)

    next_latent, reward, value, policy = world_model.recurrent_step(
        latent,
        jnp.array(0, dtype=jnp.float32),
        training=True,
    )
    assert next_latent.state.shape == latent.state.shape
    assert reward.shape == (world_model.config.support_bins,)
    assert value.shape == (world_model.config.v_num, world_model.config.support_bins)
    assert policy.shape == (world_model.env.action_dim * 2,)


def test_rejects_non_efficient_zero_config(cartpole_training_env: TrainingEnv) -> None:
    from algorl.agents.configs import BaseAgentConfig

    context = ComponentContext(
        backend=get_backend("jax"),
        config=BaseAgentConfig(),
        env=cartpole_training_env,
    )
    with pytest.raises(TypeError, match="EfficientZeroConfig"):
        build_efficient_zero_world_model(context)
