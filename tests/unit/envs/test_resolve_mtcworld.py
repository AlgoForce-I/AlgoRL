"""Resolve native MTCWorldMJX environments for agents."""

from __future__ import annotations

import pytest

pytest.importorskip("MTCWorldMJX")

from MTCWorldMJX import make, make_cl_train_env

from algorl.agents.configs import DreamerV3Config
from algorl.agents.world_model.dreamer_v3 import DreamerV3
from algorl.envs import resolve_env
from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv


def test_resolve_sawyer_xyz_env() -> None:
    env = resolve_env(make("reach-v3"), seed=0)
    assert not env.is_batched
    assert env.observation_space.shape == (39,)
    assert env.action_space.shape == (4,)
    assert isinstance(env.raw, JaxEnv)


def test_resolve_continual_learning_env() -> None:
    env = resolve_env(make_cl_train_env("CW10", steps_per_task=50), seed=0)
    assert not env.is_batched
    assert env.observation_space.shape == (49,)
    assert isinstance(env.raw, JaxEnv)


def test_resolve_vector_env() -> None:
    from MTCWorldMJX import make_mt_envs

    env = resolve_env(make_mt_envs("reach-v3", seed=0, num_envs=4), seed=0)
    assert env.is_batched
    assert env.num_envs == 4
    assert isinstance(env.raw, BatchedJaxEnv)


def test_dreamer_accepts_native_sawyer_env() -> None:
    config = DreamerV3Config(require_implemented=False)
    agent = DreamerV3(make("reach-v3"), config=config)
    assert agent.observation_space.shape == (39,)


def test_dreamer_accepts_native_continual_env() -> None:
    config = DreamerV3Config(require_implemented=False)
    agent = DreamerV3(make_cl_train_env("CW10", steps_per_task=50), config=config)
    assert agent.observation_space.shape == (49,)
