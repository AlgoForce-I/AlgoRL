"""Continual World (CW10/CW20) adapter tests (optional dependency)."""

from __future__ import annotations

import pytest

pytest.importorskip("MTCWorldMJX")

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import AlphaZeroConfig
from algorl.backends.jax.backend import JAXBackend
from algorl.backends.jax.envs import (
    BatchedContinualLearningJaxEnv,
    MtcworldContinualGymEnv,
    MtcworldContinualRolloutCollector,
    MtcworldCWEvalGymEnv,
    MtcworldCWRolloutCollector,
    MtcworldCWSearchEnvironment,
    make_batched_cw_train_env,
    search_env_from_context,
)
from algorl.core.component_context import ComponentContext
from algorl.envs import resolve_env


@pytest.fixture
def cw_train_env() -> MtcworldContinualGymEnv:
    return MtcworldContinualGymEnv("CW10", seed=0, steps_per_task=50)


@pytest.fixture
def cw_eval_env() -> MtcworldCWEvalGymEnv:
    return MtcworldCWEvalGymEnv("CW10", task_index=0, seed=0)


def test_continual_gym_env_observation_shape(cw_train_env: MtcworldContinualGymEnv) -> None:
    assert cw_train_env.num_tasks == 10
    assert cw_train_env.observation_space.shape == (49,)

    obs, info = cw_train_env.reset(seed=0)
    assert obs.shape == (49,)
    assert info["seq_idx"] == 0
    assert info["global_step"] == 0
    assert info["task_name"] == "hammer-v3"


def test_continual_gym_env_reset_preserves_cl_state(cw_train_env: MtcworldContinualGymEnv) -> None:
    cw_train_env.reset(seed=0)
    truncated = False
    info: dict[str, object] = {}
    for _ in range(cw_train_env.steps_per_task):
        _, _, _, truncated, info = cw_train_env.step(np.zeros(4, dtype=np.float32))
        if truncated:
            break

    assert truncated
    global_step = int(info["global_step"])
    seq_idx = int(info["seq_idx"])

    obs, reset_info = cw_train_env.reset()
    assert obs.shape == (49,)
    assert reset_info["global_step"] == global_step
    assert reset_info["seq_idx"] == seq_idx


def test_factory_resolves_continual_gym_env(cw_train_env: MtcworldContinualGymEnv) -> None:
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=resolve_env(cw_train_env),
    )
    search_env = search_env_from_context(context)
    assert isinstance(search_env, MtcworldCWSearchEnvironment)
    assert search_env.action_shape == (4,)


def test_cw_eval_gym_env(cw_eval_env: MtcworldCWEvalGymEnv) -> None:
    obs, info = cw_eval_env.reset(seed=0)
    assert obs.shape == (49,)
    assert info["task_index"] == 0
    assert info["task_name"] == "hammer-v3"

    next_obs, reward, terminated, truncated, step_info = cw_eval_env.step(
        np.zeros(4, dtype=np.float32)
    )
    assert next_obs.shape == (49,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "success" in step_info


def test_cw_rollout_collector() -> None:
    collector = MtcworldCWRolloutCollector("CW10", task_index=0, num_envs=4, seed=0)

    def policy(obs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        del key
        return jnp.zeros((obs.shape[0], 4), dtype=jnp.float32)

    batch = collector.collect_rollout(policy, num_steps=3)
    assert batch.num_steps == 3
    assert batch.num_envs == 4
    assert batch.observation.shape == (3, 4, 49)
    assert batch.action.shape == (3, 4, 4)


def test_continual_rollout_collector_sequence() -> None:
    collector = MtcworldContinualRolloutCollector("CW10", num_envs=2, seed=0)

    def policy(obs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        del key
        return jnp.zeros((obs.shape[0], 4), dtype=jnp.float32)

    batches = collector.collect_sequence(policy, steps_per_task=2)
    assert len(batches) == 10
    assert batches[0].observation.shape == (2, 2, 49)
    assert collector.task_names[0] == "hammer-v3"


def test_batched_continual_task_schedule() -> None:
    """Parallel lanes advance global_step; tasks switch at ``steps_per_task`` budget."""
    num_envs = 2
    steps_per_task = 4
    num_tasks = 3
    seq_idx = 0
    global_step = 0
    seen_tasks = [0]

    for _ in range(10):
        global_step += num_envs
        task_end = (seq_idx + 1) * steps_per_task
        if global_step >= task_end and seq_idx < num_tasks - 1:
            seq_idx += 1
            seen_tasks.append(seq_idx)

    assert seen_tasks == [0, 1, 2]


def test_resolve_batched_continual_learning_env() -> None:
    env = resolve_env(make_batched_cw_train_env("CW10", num_envs=4, seed=0, steps_per_task=50))
    assert env.is_batched
    assert env.num_envs == 4
    assert env.observation_space.shape == (49,)
    assert isinstance(env.raw, BatchedContinualLearningJaxEnv)
