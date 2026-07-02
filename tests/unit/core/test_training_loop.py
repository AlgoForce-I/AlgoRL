"""Training loop tests."""

from __future__ import annotations

import tempfile

import gymnasium as gym
import numpy as np
import pytest

from algorl.agents.configs import BaseAgentConfig
from algorl.buffers.efficientzero import POLICY_TARGET_INFO_KEY, SEARCH_VALUE_INFO_KEY
from algorl.buffers.replay import UniformReplayBuffer
from algorl.common.progress_bar import TqdmProgressBar
from algorl.common.tensorboard_logger import TensorboardLogger
from algorl.core.learner import Learner
from algorl.core.planner import BatchedPlanner, Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.training_loop import TrainingLoop
from algorl.core.types import Action, Observation
from algorl.envs.jax_env import JaxRolloutBatch
from algorl.envs.training_env import TrainingEnv


class _RandomPlanner(Planner):
    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space
        self.last_deterministic: bool | None = None

    def search(self, observation: Observation, **kwargs) -> Action:
        self.last_deterministic = kwargs.get("deterministic")
        return int(self.action_space.sample())


class _NoOpLearner(Learner):
    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        replay_buffer.sample(1)
        return {"loss": 0.0}


@pytest.fixture
def cartpole_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


def test_training_loop_collects_transitions(cartpole_env: TrainingEnv) -> None:
    config = BaseAgentConfig(learning_starts=0, train_freq=1, batch_size=1, seed=0)
    buffer = UniformReplayBuffer(capacity=100)
    planner = _RandomPlanner(cartpole_env.action_space)
    loop = TrainingLoop(
        env=cartpole_env,
        planner=planner,
        learner=_NoOpLearner(),
        replay_buffer=buffer,
        config=config,
    )
    loop.run(5)
    assert len(buffer) == 5
    assert loop.logger.history[-1]["step"] == 4
    assert planner.last_deterministic is False


def test_training_loop_waits_for_learning_starts(cartpole_env: TrainingEnv) -> None:
    config = BaseAgentConfig(learning_starts=10, train_freq=1, batch_size=1, seed=0)
    buffer = UniformReplayBuffer(capacity=100)

    class _CountingLearner(Learner):
        def __init__(self) -> None:
            self.calls = 0

        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            self.calls += 1
            return {"loss": float(self.calls)}

    learner = _CountingLearner()
    loop = TrainingLoop(
        env=cartpole_env,
        planner=_RandomPlanner(cartpole_env.action_space),
        learner=learner,
        replay_buffer=buffer,
        config=config,
    )
    loop.run(8)
    assert learner.calls == 0
    assert all("loss" not in entry or entry.get("train/loss", 0) == 0 for entry in loop.logger.history)


def test_training_loop_logs_episode_metrics_with_tensorboard(cartpole_env: TrainingEnv) -> None:
    config = BaseAgentConfig(learning_starts=10, train_freq=1, batch_size=1, seed=0)
    buffer = UniformReplayBuffer(capacity=100)

    class _ZeroRewardEnv(TrainingEnv):
        def __init__(self, base: TrainingEnv) -> None:
            self._base = base

        def reset(self, *, seed: int | None = None):
            return self._base.reset(seed=seed)

        def step(self, action):
            obs, _, terminated, truncated, info = self._base.step(action)
            return obs, 1.0, terminated, truncated, info

        @property
        def observation_space(self):
            return self._base.observation_space

        @property
        def action_space(self):
            return self._base.action_space

        @property
        def raw(self):
            return self._base.raw

        @property
        def is_batched(self):
            return self._base.is_batched

    with tempfile.TemporaryDirectory() as log_dir:
        logger = TensorboardLogger(log_dir)
        env = _ZeroRewardEnv(cartpole_env)
        loop = TrainingLoop(
            env=env,
            planner=_RandomPlanner(cartpole_env.action_space),
            learner=_NoOpLearner(),
            replay_buffer=buffer,
            config=config,
            logger=logger,
        )
        loop.run(200)

        episode_entries = [
            entry
            for entry in logger.history
            if "train/episode_return" in entry
        ]
        assert episode_entries
        assert episode_entries[0]["train/episode_return"] > 0.0


def test_training_loop_progress_bar(cartpole_env: TrainingEnv) -> None:
    config = BaseAgentConfig(learning_starts=0, train_freq=1, batch_size=1, seed=0)
    buffer = UniformReplayBuffer(capacity=100)
    progress_bar = TqdmProgressBar(disable=True)
    loop = TrainingLoop(
        env=cartpole_env,
        planner=_RandomPlanner(cartpole_env.action_space),
        learner=_NoOpLearner(),
        replay_buffer=buffer,
        config=config,
    )
    loop.run(5, progress_bar=progress_bar)
    assert len(buffer) == 5


class _BatchedSearchResult:
    def __init__(self, batch_size: int, action_dim: int, num_candidates: int) -> None:
        self.batch_size = batch_size
        self.action_weights = np.ones((batch_size, num_candidates), dtype=np.float32)
        self.root_values = np.arange(batch_size, dtype=np.float32)
        self.root_candidates = np.zeros((batch_size, num_candidates, action_dim), dtype=np.float32)
        self.actions = np.zeros((batch_size, action_dim), dtype=np.float32)
        self.pred_values = np.arange(batch_size, dtype=np.float32)


class _BatchedPlanner(BatchedPlanner):
    def __init__(self, batch_size: int, action_dim: int = 1) -> None:
        self.last_result = _BatchedSearchResult(batch_size, action_dim, num_candidates=2)

    def search(self, observation: Observation, **kwargs) -> Action:
        del observation, kwargs
        return 0

    def search_batch(self, observations, **kwargs):
        del kwargs
        batch_size = len(observations)
        self.last_result = _BatchedSearchResult(batch_size, action_dim=1, num_candidates=2)
        return self.last_result


class _MockBatchedEnv(TrainingEnv):
    def __init__(self, *, num_envs: int = 2) -> None:
        super().__init__(
            observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            raw=object(),
            is_batched=True,
            num_envs=num_envs,
        )

    def collect_rollout(self, policy, num_steps: int, *, key=None) -> JaxRolloutBatch:
        del key
        observations = np.zeros((num_steps, self.num_envs, 3), dtype=np.float32)
        actions = np.zeros((num_steps, self.num_envs, 1), dtype=np.float32)
        for step_idx in range(num_steps):
            actions[step_idx] = np.asarray(
                policy(observations[step_idx], np.array(0)),
                dtype=np.float32,
            )
        return JaxRolloutBatch(
            observation=observations,
            action=actions,
            reward=np.zeros((num_steps, self.num_envs), dtype=np.float32),
            next_observation=observations,
            done=np.zeros((num_steps, self.num_envs), dtype=bool),
        )


def test_training_loop_batched_rollout_stores_search_targets() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)
    buffer = UniformReplayBuffer(capacity=100)
    planner = _BatchedPlanner(batch_size=num_envs)
    config = BaseAgentConfig(learning_starts=0, train_freq=100, batch_size=100, seed=0, jax_rollout_chunk=2)
    loop = TrainingLoop(
        env=env,
        planner=planner,
        learner=_NoOpLearner(),
        replay_buffer=buffer,
        config=config,
    )
    loop.run(4)
    assert len(buffer) == 4
    stored = tuple(buffer._storage)
    for index, transition in enumerate(stored):
        assert POLICY_TARGET_INFO_KEY in transition.info
        assert SEARCH_VALUE_INFO_KEY in transition.info
        assert transition.info[SEARCH_VALUE_INFO_KEY] == float(index % num_envs)


def test_training_loop_merges_rollout_step_info() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)
    buffer = UniformReplayBuffer(capacity=100)
    planner = _BatchedPlanner(batch_size=num_envs)
    config = BaseAgentConfig(learning_starts=0, train_freq=100, batch_size=100, seed=0, jax_rollout_chunk=1)
    loop = TrainingLoop(
        env=env,
        planner=planner,
        learner=_NoOpLearner(),
        replay_buffer=buffer,
        config=config,
    )

    original_collect = env.collect_rollout

    def collect_with_info(policy, num_steps, *, key=None):
        batch = original_collect(policy, num_steps, key=key)
        return JaxRolloutBatch(
            observation=batch.observation,
            action=batch.action,
            reward=batch.reward,
            next_observation=batch.next_observation,
            done=batch.done,
            step_info=[
                [
                    {"seq_idx": 0, "task_name": "hammer-v3", "success": 0.0},
                    {"seq_idx": 0, "task_name": "hammer-v3", "success": 1.0},
                ]
            ],
        )

    env.collect_rollout = collect_with_info
    loop.run(2)
    stored = tuple(buffer._storage)
    assert stored[0].info["task_name"] == "hammer-v3"
    assert stored[1].info["success"] == 1.0
    assert POLICY_TARGET_INFO_KEY in stored[0].info
