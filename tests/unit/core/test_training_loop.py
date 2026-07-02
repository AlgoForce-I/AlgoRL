"""Training loop tests."""

from __future__ import annotations

import tempfile

import gymnasium as gym
import pytest

from algorl.agents.configs import BaseAgentConfig
from algorl.buffers.replay import UniformReplayBuffer
from algorl.common.tensorboard_logger import TensorboardLogger
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.training_loop import TrainingLoop
from algorl.core.types import Action, Observation
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
