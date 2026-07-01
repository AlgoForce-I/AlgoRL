"""MTCWorldMJX adapter tests (optional dependency)."""

from __future__ import annotations

import pytest

pytest.importorskip("MTCWorldMJX")

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import AlphaZeroConfig
from algorl.backends.jax.backend import JAXBackend
from algorl.backends.jax.envs import (
    MtcworldGymEnv,
    MtcworldRolloutCollector,
    MtcworldSearchEnvironment,
    search_env_from_context,
)
from algorl.core.component_context import ComponentContext
from algorl.envs import resolve_env


@pytest.fixture
def reach_env() -> MtcworldGymEnv:
    return MtcworldGymEnv("reach-v3", seed=0)


def test_mtcworld_gym_env_reset_and_step(reach_env: MtcworldGymEnv) -> None:
    obs, info = reach_env.reset(seed=0)
    assert obs.shape == reach_env.observation_space.shape
    assert info == {}

    action = reach_env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = reach_env.step(action)
    assert next_obs.shape == obs.shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_factory_resolves_mtcworld_gym_env(reach_env: MtcworldGymEnv) -> None:
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=resolve_env(reach_env),
    )
    search_env = search_env_from_context(context)
    assert isinstance(search_env, MtcworldSearchEnvironment)
    assert not search_env.is_discrete
    assert search_env.action_shape == (4,)


def test_mtcworld_search_environment_step(reach_env: MtcworldGymEnv) -> None:
    obs, _ = reach_env.reset(seed=0)
    search_env = reach_env.search_environment()
    state = search_env.initial_state(obs)

    next_state, reward = search_env.step(state, jnp.zeros(4, dtype=jnp.float32))
    assert jnp.isfinite(reward)
    assert search_env.canonical_observation(next_state).shape == (39,)
    assert search_env.invalid_actions(next_state) is None


def test_mtcworld_rollout_collector(reach_env: MtcworldGymEnv) -> None:
    del reach_env
    collector = MtcworldRolloutCollector("reach-v3", num_envs=4, seed=0)

    def policy(obs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        del key
        return jnp.zeros((obs.shape[0], 4), dtype=jnp.float32)

    batch = collector.collect_rollout(policy, num_steps=3)
    assert batch.num_steps == 3
    assert batch.num_envs == 4
    assert batch.observation.shape == (3, 4, 39)
    assert batch.action.shape == (3, 4, 4)
    assert batch.reward.shape == (3, 4)
    assert batch.done.shape == (3, 4)

    transitions = batch.to_transitions()
    assert len(transitions) == 12
