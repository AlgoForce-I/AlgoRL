"""Tests for Gymnasium vector batched env adapter."""

from __future__ import annotations

import gymnasium as gym
import jax
import numpy as np
import pytest

from algorl.backends.jax.envs import GymnasiumVectorJaxEnv, make_gymnasium_vector_env
from algorl.envs import resolve_env
from algorl.envs.jax_env import BatchedJaxEnv


def test_make_gymnasium_vector_env_shapes() -> None:
    jax_env = make_gymnasium_vector_env(lambda: gym.make("CartPole-v1"), num_envs=4, seed=0)
    assert jax_env.num_envs == 4
    assert jax_env.observation_shape == (4,)
    assert jax_env.action_shape == ()


def test_resolve_gymnasium_vector_env() -> None:
    vector_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(3)])
    training_env = resolve_env(vector_env, seed=0)
    assert training_env.is_batched
    assert training_env.num_envs == 3
    assert isinstance(training_env.raw, BatchedJaxEnv)


def test_collect_rollout_persists_state_across_calls() -> None:
    def policy(observations: np.ndarray, key: jax.Array) -> np.ndarray:
        del key
        return np.zeros((observations.shape[0],), dtype=np.int64)

    continuous = make_gymnasium_vector_env(lambda: gym.make("CartPole-v1"), num_envs=2, seed=0)
    key = jax.random.PRNGKey(0)
    key, first_key = jax.random.split(key)
    key, second_key = jax.random.split(key)
    split_first = continuous.collect_rollout(policy, 3, key=first_key)
    split_second = continuous.collect_rollout(policy, 2, key=second_key)
    split_obs = np.concatenate([split_first.observation, split_second.observation], axis=0)

    reference = make_gymnasium_vector_env(lambda: gym.make("CartPole-v1"), num_envs=2, seed=0)
    reference_batch = reference.collect_rollout(policy, 5, key=jax.random.PRNGKey(0))

    assert split_obs.shape == reference_batch.observation.shape
    assert np.allclose(split_obs, reference_batch.observation)


def test_collect_rollout_uses_final_observation_on_done() -> None:
    class _TerminalAfterOneStep(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Discrete(2)
            self._step_count = 0

        def reset(self, *, seed: int | None = None, options=None):
            del options
            if seed is not None:
                super().reset(seed=seed)
            self._step_count = 0
            return np.array([1.0, 2.0], dtype=np.float32), {}

        def step(self, action):
            del action
            self._step_count += 1
            if self._step_count == 1:
                return (
                    np.array([9.0, 9.0], dtype=np.float32),
                    1.0,
                    True,
                    False,
                    {},
                )
            raise RuntimeError("env should auto-reset before a second step")

    vector_env = gym.vector.SyncVectorEnv(
        [_TerminalAfterOneStep for _ in range(1)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    jax_env = GymnasiumVectorJaxEnv(vector_env, seed=0)

    def policy(observations: np.ndarray, key: jax.Array) -> np.ndarray:
        del observations, key
        return np.zeros((1,), dtype=np.int64)

    batch = jax_env.collect_rollout(policy, 1, key=jax.random.PRNGKey(0))
    assert batch.done[0, 0]
    assert np.allclose(batch.next_observation[0, 0], np.array([9.0, 9.0], dtype=np.float32))


class _TerminalAfterTwoSteps(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._step_count = 0

    def reset(self, *, seed: int | None = None, options=None):
        del options
        if seed is not None:
            super().reset(seed=seed)
        self._step_count = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        del action
        self._step_count += 1
        obs = np.array([float(self._step_count), 0.0], dtype=np.float32)
        return obs, 1.0, self._step_count >= 2, False, {}


def test_collect_rollout_flags_next_step_autoreset_transitions() -> None:
    vector_env = gym.vector.SyncVectorEnv([_TerminalAfterTwoSteps for _ in range(1)])
    assert vector_env.metadata["autoreset_mode"] == gym.vector.AutoresetMode.NEXT_STEP
    jax_env = GymnasiumVectorJaxEnv(vector_env, seed=0)

    def policy(observations: np.ndarray, key: jax.Array) -> np.ndarray:
        del observations, key
        return np.zeros((1,), dtype=np.int64)

    batch = jax_env.collect_rollout(policy, 4, key=jax.random.PRNGKey(0))
    assert batch.step_info is not None
    dones = batch.done[:, 0].tolist()
    skips = [bool(info[0].get("replay_skip", False)) for info in batch.step_info]
    # Episode ends at step 1; step 2 is the reset filler and must be flagged.
    assert dones == [False, True, False, False]
    assert skips == [False, False, True, False]
    # The filler step carries the fabricated zero reward.
    assert batch.reward[2, 0] == 0.0


def test_collect_rollout_flags_autoreset_across_chunk_boundary() -> None:
    vector_env = gym.vector.SyncVectorEnv([_TerminalAfterTwoSteps for _ in range(1)])
    jax_env = GymnasiumVectorJaxEnv(vector_env, seed=0)

    def policy(observations: np.ndarray, key: jax.Array) -> np.ndarray:
        del observations, key
        return np.zeros((1,), dtype=np.int64)

    first = jax_env.collect_rollout(policy, 2, key=jax.random.PRNGKey(0))
    second = jax_env.collect_rollout(policy, 2, key=jax.random.PRNGKey(1))
    assert bool(first.done[1, 0])
    assert second.step_info is not None
    assert bool(second.step_info[0][0].get("replay_skip", False))
    assert not second.step_info[1][0].get("replay_skip", False)


def test_collect_rollout_continuous_action_shape() -> None:
    pytest.importorskip("mujoco")
    jax_env = make_gymnasium_vector_env(lambda: gym.make("HalfCheetah-v5"), num_envs=2, seed=0)

    def policy(observations: np.ndarray, key: jax.Array) -> np.ndarray:
        del key
        return np.zeros((observations.shape[0], 6), dtype=np.float32)

    batch = jax_env.collect_rollout(policy, 1, key=jax.random.PRNGKey(0))
    assert batch.action.shape == (1, 2, 6)
    assert batch.observation.shape[1:] == (2, 17)
