"""Gymnasium SearchEnvironment adapter tests."""

from __future__ import annotations

import gymnasium as gym
import jax.numpy as jnp
import pytest

from algorl.agents.configs import AlphaZeroConfig
from algorl.backends.jax.backend import JAXBackend
from algorl.backends.jax.envs import GymnasiumSearchEnvironment, search_env_from_context
from algorl.backends.jax.planners.mcts.alphazero import AlphaZeroPlanner
from algorl.core.component_context import ComponentContext


def test_factory_resolves_discrete_gym_env(cartpole_env: gym.Env) -> None:
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=cartpole_env,
    )
    search_env = search_env_from_context(context)
    assert isinstance(search_env, GymnasiumSearchEnvironment)
    assert search_env.is_discrete


def test_factory_resolves_continuous_box_env() -> None:
    env = gym.make("MountainCarContinuous-v0")
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=env,
    )
    search_env = search_env_from_context(context)
    assert isinstance(search_env, GymnasiumSearchEnvironment)
    assert not search_env.is_discrete
    assert search_env.action_shape == (1,)
    env.close()


def test_factory_rejects_unsupported_action_space() -> None:
    class _MultiDiscreteEnv(gym.Env):
        action_space = gym.spaces.MultiDiscrete([2, 2])
        observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            raise NotImplementedError

    env = _MultiDiscreteEnv()
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=env,
    )
    with pytest.raises(NotImplementedError, match="No JAX SearchEnvironment adapter"):
        search_env_from_context(context)


def test_gymnasium_search_environment_step_discrete(cartpole_env: gym.Env) -> None:
    cartpole_env.reset(seed=0)
    search_env = GymnasiumSearchEnvironment(cartpole_env)
    obs, _ = cartpole_env.reset(seed=0)

    state = search_env.initial_state(obs)
    next_state, reward = search_env.step(state, jnp.int32(1))

    assert next_state.observation.shape == state.observation.shape
    assert float(reward) != 0.0 or bool(next_state.terminated) or bool(next_state.truncated)


def test_gymnasium_search_environment_step_continuous() -> None:
    env = gym.make("MountainCarContinuous-v0")
    env.reset(seed=0)
    search_env = GymnasiumSearchEnvironment(env)
    obs, _ = env.reset(seed=0)

    state = search_env.initial_state(obs)
    assert search_env.invalid_actions(state) is None

    next_state, reward = search_env.step(state, jnp.array([0.5], dtype=jnp.float32))
    assert next_state.observation.shape == state.observation.shape
    assert jnp.isfinite(reward)
    env.close()


def test_alphazero_search_on_cartpole(cartpole_env: gym.Env) -> None:
    def fake_evaluate(params, observations):
        batch_size = observations.shape[0]
        num_actions = params["num_actions"]
        return (
            jnp.zeros((batch_size, num_actions), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.float32),
        )

    cartpole_env.reset(seed=0)
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False, mcts_simulations=4),
        env=cartpole_env,
    )
    planner = AlphaZeroPlanner(context, evaluate=fake_evaluate)
    planner.params = {"num_actions": cartpole_env.action_space.n}

    obs, _ = cartpole_env.reset(seed=0)
    result = planner.search_batch(obs, deterministic=True)
    assert result.batch_size == 1
