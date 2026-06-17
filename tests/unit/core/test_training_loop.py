"""Training loop tests."""

from __future__ import annotations

import gymnasium as gym
import pytest

from algorl.agents.configs import BaseAgentConfig
from algorl.buffers.replay import UniformReplayBuffer
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.training_loop import TrainingLoop
from algorl.core.types import Action, Observation


class _RandomPlanner(Planner):
    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space

    def search(self, observation: Observation, **kwargs) -> Action:
        return int(self.action_space.sample())


class _NoOpLearner(Learner):
    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        replay_buffer.sample(1)
        return {"loss": 0.0}


@pytest.fixture
def cartpole_env() -> gym.Env:
    return gym.make("CartPole-v1")


def test_training_loop_collects_transitions(cartpole_env: gym.Env) -> None:
    config = BaseAgentConfig(learning_starts=0, train_freq=1, batch_size=1, seed=0)
    buffer = UniformReplayBuffer(capacity=100)
    loop = TrainingLoop(
        env=cartpole_env,
        planner=_RandomPlanner(cartpole_env.action_space),
        learner=_NoOpLearner(),
        replay_buffer=buffer,
        config=config,
    )
    loop.run(5)
    assert len(buffer) == 5
    assert loop.logger.history[-1]["step"] == 4


def test_training_loop_waits_for_learning_starts(cartpole_env: gym.Env) -> None:
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
    assert all("loss" not in entry or entry.get("loss", 0) == 0 for entry in loop.logger.history)
