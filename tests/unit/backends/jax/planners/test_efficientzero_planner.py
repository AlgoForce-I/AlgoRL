"""EfficientZero planner tests."""

from __future__ import annotations

import gymnasium as gym
import jax.numpy as jnp
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.planners.efficientzero import (
    EfficientZeroPlanner,
    build_efficient_zero_planner,
    continuous_search_config_from_agent,
    uses_continuous_search,
)
from algorl.backends.jax.world_models.efficientzero import build_efficient_zero_world_model
from algorl.core.component_context import ComponentContext
from algorl.core.factory import get_backend
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_training_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


@pytest.fixture
def planner_context(cartpole_training_env: TrainingEnv) -> ComponentContext:
    config = EfficientZeroConfig.for_dmc_state(
        require_implemented=False,
        mcts_simulations=4,
        search_batch_size=1,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    context.world_model = build_efficient_zero_world_model(context)
    return context


def test_continuous_search_config_from_agent_maps_simulations() -> None:
    dmc_config = EfficientZeroConfig.for_dmc_state(mcts_simulations=24, gumbel_scale=0.5)
    dmc_search = continuous_search_config_from_agent(dmc_config)
    assert dmc_search.num_simulations == 24
    assert dmc_search.use_gumbel_noise is False
    assert dmc_search.gumbel_scale == 0.0

    atari_config = EfficientZeroConfig.for_atari(mcts_simulations=16, gumbel_scale=0.75)
    atari_search = continuous_search_config_from_agent(atari_config)
    assert atari_search.num_simulations == 16
    assert atari_search.use_gumbel_noise is False
    assert atari_search.gumbel_scale == 0.0


def test_uses_continuous_search_routing() -> None:
    dmc_config = EfficientZeroConfig.for_dmc_state()
    assert uses_continuous_search(dmc_config, resolved_model_type="dmc_state") is True
    assert uses_continuous_search(dmc_config, resolved_model_type="dmc_image") is True

    atari_config = EfficientZeroConfig.for_atari()
    assert uses_continuous_search(atari_config, resolved_model_type="atari") is False

    discrete_dmc = EfficientZeroConfig.for_dmc_state(policy_distribution="discrete")
    assert uses_continuous_search(discrete_dmc, resolved_model_type="dmc_state") is False


def test_efficient_zero_planner_rejects_atari_discrete_search(
    planner_context: ComponentContext,
) -> None:
    planner = EfficientZeroPlanner(planner_context)
    planner.config = EfficientZeroConfig.for_atari(require_implemented=False, mcts_simulations=4)
    with pytest.raises(NotImplementedError, match="discrete Gumbel search"):
        planner._ensure_continuous_search_supported()


def test_build_efficient_zero_planner_factory(planner_context: ComponentContext) -> None:
    planner = build_efficient_zero_planner(planner_context)
    assert isinstance(planner, EfficientZeroPlanner)
    assert planner.params is planner_context.world_model.params


def test_efficient_zero_planner_search_cartpole(planner_context: ComponentContext) -> None:
    planner = EfficientZeroPlanner(planner_context)
    obs = jnp.zeros((4,), dtype=jnp.float32)
    action = planner.search(obs, deterministic=True)
    assert jnp.asarray(action).shape == (1,)

    result = planner.search_batch(obs, deterministic=True)
    assert result.actions.shape == (1, 1)
    assert result.action_weights.shape == (1, planner.search_config.num_sampled_actions)
    assert jnp.isfinite(result.root_values).all()
