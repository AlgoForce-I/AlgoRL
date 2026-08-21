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

    def collect_rollout(self, policy, num_steps: int, *, key=None, on_step=None) -> JaxRolloutBatch:
        del key
        observations = np.zeros((num_steps, self.num_envs, 3), dtype=np.float32)
        actions = np.zeros((num_steps, self.num_envs, 1), dtype=np.float32)
        for step_idx in range(num_steps):
            actions[step_idx] = np.asarray(
                policy(observations[step_idx], np.array(0)),
                dtype=np.float32,
            )
            if on_step is not None:
                on_step(
                    self.num_envs,
                    {
                        "train/reward": 0.0,
                        "rewards": np.zeros(self.num_envs, dtype=np.float32),
                        "dones": np.zeros(self.num_envs, dtype=bool),
                        "infos": [{} for _ in range(self.num_envs)],
                    },
                )
        return JaxRolloutBatch(
            observation=observations,
            action=actions,
            reward=np.zeros((num_steps, self.num_envs), dtype=np.float32),
            next_observation=observations,
            done=np.zeros((num_steps, self.num_envs), dtype=bool),
        )


def test_batched_gradient_steps_per_rollout_limits_train_calls() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)

    class _CountingLearner(Learner):
        def __init__(self) -> None:
            self.calls = 0

        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            self.calls += 1
            return {"loss": float(self.calls)}

        def train_burst(self, replay_buffer: ReplayBuffer, steps: int, **kwargs) -> dict[str, float]:
            del replay_buffer, kwargs
            self.calls += steps
            return {"loss": float(self.calls)}

    learner = _CountingLearner()
    buffer = UniformReplayBuffer(capacity=100)
    config = BaseAgentConfig(
        learning_starts=0,
        train_freq=1,
        batch_size=1,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=1,
    )
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=learner,
        replay_buffer=buffer,
        config=config,
    )
    loop.run(8)
    assert len(buffer) == 8
    assert learner.calls == 2


def test_training_loop_batched_train_burst_runs_capped_updates() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)

    class _BurstLearner(Learner):
        def __init__(self) -> None:
            self.burst_sizes: list[int] = []

        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            del replay_buffer
            return {"loss": 0.0}

        def train_burst(self, replay_buffer: ReplayBuffer, steps: int, **kwargs) -> dict[str, float]:
            del replay_buffer, kwargs
            self.burst_sizes.append(steps)
            return {"loss": float(steps)}

    learner = _BurstLearner()
    buffer = UniformReplayBuffer(capacity=100)
    config = BaseAgentConfig(
        learning_starts=0,
        train_freq=1,
        batch_size=1,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=num_envs,
    )
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=learner,
        replay_buffer=buffer,
        config=config,
    )
    loop.run(8)
    assert learner.burst_sizes == [num_envs, num_envs]


def test_training_loop_syncs_self_play_before_rollout_when_enabled() -> None:
    from algorl.agents.configs import EfficientZeroConfig

    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)

    class _SyncLearner(Learner):
        def __init__(self) -> None:
            self.sync_calls = 0

        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            del replay_buffer
            return {"loss": 0.0}

        def train_burst(self, replay_buffer: ReplayBuffer, steps: int, **kwargs) -> dict[str, float]:
            del replay_buffer, kwargs, steps
            return {"loss": 0.0}

        def sync_self_play_for_rollout(self) -> None:
            self.sync_calls += 1

    learner = _SyncLearner()
    buffer = UniformReplayBuffer(capacity=100)
    config = EfficientZeroConfig.for_batched(num_envs=num_envs).with_overrides(
        learning_starts=0,
        train_freq=1,
        batch_size=1,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=num_envs,
        sync_self_play_before_rollout=True,
    )
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=learner,
        replay_buffer=buffer,
        config=config,
    )
    loop.run(8)
    # One sync per rollout chunk (8 env-steps / (2 envs * 2 chunk steps) = 2 chunks).
    assert learner.sync_calls == 2

    learner_off = _SyncLearner()
    config_off = config.with_overrides(sync_self_play_before_rollout=False)
    loop_off = TrainingLoop(
        env=_MockBatchedEnv(num_envs=num_envs),
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=learner_off,
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=config_off,
    )
    loop_off.run(8)
    assert learner_off.sync_calls == 0


def test_training_loop_batched_interleaves_training_with_replay_buffer_warmup() -> None:
    """Batched mode should match sequential timing once buffer size gates training.

    With num_envs=2 and jax_rollout_chunk=2, the first chunk adds 4 transitions.
    If batch_size=3 and learning_starts=0, sequential would start training on
    global step 2 (len(buffer) becomes 3) and then again on step 3.
    """
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)

    class _CountingLearner(Learner):
        def __init__(self) -> None:
            self.calls = 0

        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            # Ensure train is only called when the buffer is warm enough.
            assert len(replay_buffer) >= 3
            self.calls += 1
            return {"loss": float(self.calls)}

    learner = _CountingLearner()
    buffer = UniformReplayBuffer(capacity=100)
    config = BaseAgentConfig(
        learning_starts=0,
        train_freq=1,
        batch_size=3,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=None,
    )
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=learner,
        replay_buffer=buffer,
        config=config,
    )
    loop.run(4)
    assert len(buffer) == 4
    assert learner.calls == 2


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


def test_training_loop_batched_transitions_tag_env_id_and_skip_autoreset_steps() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)
    planner = _BatchedPlanner(batch_size=num_envs)
    loop = TrainingLoop(
        env=env,
        planner=planner,
        learner=_NoOpLearner(),
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=BaseAgentConfig(learning_starts=0, train_freq=100, batch_size=100, seed=0),
    )

    num_steps = 2
    observations = np.zeros((num_steps, num_envs, 3), dtype=np.float32)
    batch = JaxRolloutBatch(
        observation=observations,
        action=np.zeros((num_steps, num_envs, 1), dtype=np.float32),
        reward=np.zeros((num_steps, num_envs), dtype=np.float32),
        next_observation=observations,
        done=np.zeros((num_steps, num_envs), dtype=bool),
        step_info=[
            [{}, {}],
            # Env 1's second step is a NEXT_STEP autoreset filler.
            [{}, {"replay_skip": True}],
        ],
    )
    transitions = loop._transitions_from_rollout(batch, search_results=[])

    assert len(transitions) == num_steps * num_envs - 1
    assert [transition.info["env_id"] for transition in transitions] == [0, 1, 0]


def test_training_loop_progress_bar_shows_training_phase(cartpole_env: TrainingEnv) -> None:
    config = BaseAgentConfig(learning_starts=0, train_freq=1, batch_size=1, seed=0)
    buffer = UniformReplayBuffer(capacity=100)

    class _LossLearner(Learner):
        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            replay_buffer.sample(1)
            return {"loss": 0.42, "policy_loss": 0.1}

    class _TrackingProgressBar:
        def __init__(self) -> None:
            self.pulses: list[dict[str, object]] = []
            self.n = 0

        def start(self, total: int, **kwargs) -> None:
            del total, kwargs

        def update(self, step_info=None, *, n: int = 1) -> None:
            del step_info
            self.n += n

        def pulse(self, step_info=None) -> None:
            if step_info is not None:
                self.pulses.append(dict(step_info))

        def close(self) -> None:
            return

    progress_bar = _TrackingProgressBar()
    loop = TrainingLoop(
        env=cartpole_env,
        planner=_RandomPlanner(cartpole_env.action_space),
        learner=_LossLearner(),
        replay_buffer=buffer,
        config=config,
    )
    loop.run(3, progress_bar=progress_bar)
    phases = [pulse.get("phase") for pulse in progress_bar.pulses]
    assert "training" in phases
    assert any(pulse.get("loss") == 0.42 for pulse in progress_bar.pulses)


def test_training_loop_batched_tensorboard_logs_during_rollout() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)
    buffer = UniformReplayBuffer(capacity=100)
    planner = _BatchedPlanner(batch_size=num_envs)
    config = BaseAgentConfig(learning_starts=0, train_freq=100, batch_size=100, seed=0, jax_rollout_chunk=2)
    with tempfile.TemporaryDirectory() as log_dir:
        logger = TensorboardLogger(log_dir)
        loop = TrainingLoop(
            env=env,
            planner=planner,
            learner=_NoOpLearner(),
            replay_buffer=buffer,
            config=config,
            logger=logger,
        )
        loop.run(4)
        assert len(buffer) == 4
        assert any("train/batched/mean_reward" in entry for entry in logger.history)
        assert any("train/reward" in entry for entry in logger.history)


def test_training_loop_batched_logs_per_lane_episode_returns() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)
    buffer = UniformReplayBuffer(capacity=100)
    planner = _BatchedPlanner(batch_size=num_envs)
    config = BaseAgentConfig(learning_starts=0, train_freq=100, batch_size=100, seed=0, jax_rollout_chunk=1)

    original_collect = env.collect_rollout

    def collect_with_done(policy, num_steps, *, key=None, on_step=None):
        batch = original_collect(policy, num_steps, key=key, on_step=None)
        if on_step is not None:
            on_step(
                num_envs,
                {
                    "train/reward": 0.0,
                    "rewards": np.asarray([1.0, 10.0], dtype=np.float32),
                    "dones": np.asarray([True, False], dtype=bool),
                    "infos": [{}, {}],
                },
            )
        return batch

    env.collect_rollout = collect_with_done

    with tempfile.TemporaryDirectory() as log_dir:
        logger = TensorboardLogger(log_dir)
        loop = TrainingLoop(
            env=env,
            planner=planner,
            learner=_NoOpLearner(),
            replay_buffer=buffer,
            config=config,
            logger=logger,
        )
        loop.run(2)
        episode_returns = [
            entry["train/episode_return"]
            for entry in logger.history
            if "train/episode_return" in entry
        ]
        assert episode_returns
        assert all(return_value == 1.0 for return_value in episode_returns)
        assert 11.0 not in episode_returns
        summary_entries = [
            entry for entry in logger.history if "train/batched/mean_episode_return" in entry
        ]
        assert summary_entries
        assert summary_entries[0]["train/batched/mean_episode_return"] == 1.0


def test_training_loop_batched_progress_bar_updates_during_rollout() -> None:
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

    class _CountingProgressBar:
        def __init__(self) -> None:
            self.total = 0
            self.n = 0
            self.pulses: list[dict[str, object]] = []

        def start(self, total: int, **kwargs) -> None:
            del kwargs
            self.total = total

        def update(self, step_info=None, *, n: int = 1) -> None:
            del step_info
            self.n += n

        def pulse(self, step_info=None) -> None:
            if step_info is not None:
                self.pulses.append(dict(step_info))

        def close(self) -> None:
            return

    progress_bar = _CountingProgressBar()
    loop.run(4, progress_bar=progress_bar)
    assert progress_bar.n == 4
    assert len(buffer) == 4
    assert any(pulse.get("phase") == "search" for pulse in progress_bar.pulses)


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

    def collect_with_info(policy, num_steps, *, key=None, on_step=None):
        batch = original_collect(policy, num_steps, key=key, on_step=on_step)
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


def test_training_loop_notifies_learner_on_task_boundary() -> None:
    num_envs = 1

    class _TaskBoundaryEnv(_MockBatchedEnv):
        def __init__(self, *, num_envs: int = 2) -> None:
            super().__init__(num_envs=num_envs)
            self._sent_boundary = False

        def collect_rollout(self, policy, num_steps: int, *, key=None, on_step=None):
            observations = np.zeros((num_steps, self.num_envs, 3), dtype=np.float32)
            actions = np.zeros((num_steps, self.num_envs, 1), dtype=np.float32)
            step_info = []
            for step_idx in range(num_steps):
                actions[step_idx] = np.asarray(
                    policy(observations[step_idx], np.array(0)),
                    dtype=np.float32,
                )
                task_changed = step_idx == 0 and not self._sent_boundary
                if task_changed:
                    self._sent_boundary = True
                info = {"seq_idx": 0, "task_changed": task_changed}
                step_info.append([info])
                if on_step is not None:
                    on_step(
                        self.num_envs,
                        {
                            "train/reward": 0.0,
                            "rewards": np.zeros(self.num_envs, dtype=np.float32),
                            "dones": np.zeros(self.num_envs, dtype=bool),
                            "infos": [info],
                        },
                    )
            return JaxRolloutBatch(
                observation=observations,
                action=actions,
                reward=np.zeros((num_steps, self.num_envs), dtype=np.float32),
                next_observation=observations,
                done=np.zeros((num_steps, self.num_envs), dtype=bool),
                step_info=step_info,
            )

    class _BoundaryLearner(Learner):
        def __init__(self) -> None:
            self.boundary_task_ids: list[int] = []

        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            del replay_buffer
            return {}

        def on_task_boundary(self, new_task_id: int) -> None:
            self.boundary_task_ids.append(new_task_id)

    class _ClearingBuffer(UniformReplayBuffer):
        def clear(self) -> None:
            self._storage.clear()

    env = _TaskBoundaryEnv(num_envs=num_envs)
    buffer = _ClearingBuffer(capacity=100)
    learner = _BoundaryLearner()
    config = BaseAgentConfig(
        learning_starts=0,
        train_freq=100,
        batch_size=1,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=1,
    )
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=learner,
        replay_buffer=buffer,
        config=config,
    )
    loop.run(4)
    assert learner.boundary_task_ids == [1]
    assert len(buffer) == 3
    assert loop._min_train_step == 1 + config.learning_starts


def test_training_loop_syncs_task_id_from_env_current_task_index() -> None:
    num_envs = 1

    class _TaskIndexedEnv(_MockBatchedEnv):
        def __init__(self, *, num_envs: int = 1) -> None:
            super().__init__(num_envs=num_envs)
            self._seq_idx = 2

        @property
        def current_task_index(self) -> int:
            return self._seq_idx

    class _TaskLearner(Learner):
        def __init__(self) -> None:
            self.task_id = 0
            self.boundary_calls: list[int] = []

        def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
            del replay_buffer
            return {}

        def on_task_boundary(self, new_task_id: int) -> None:
            self.boundary_calls.append(new_task_id)
            self.task_id = int(new_task_id)

        def set_task_id(self, task_id: int) -> None:
            self.task_id = int(task_id)

    env = _TaskIndexedEnv(num_envs=num_envs)
    learner = _TaskLearner()
    config = BaseAgentConfig(
        learning_starts=0,
        train_freq=100,
        batch_size=1,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=1,
    )
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=learner,
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=config,
    )
    loop.run(2)
    assert learner.boundary_calls == [2]
    assert learner.task_id == 2


class _EvalCountingEnv:
    """Wrap a sequential env and count train ``step`` calls."""

    def __init__(self, base: TrainingEnv) -> None:
        self._base = base
        self.step_calls = 0

    def reset(self, *, seed: int | None = None):
        return self._base.reset(seed=seed)

    def step(self, action):
        self.step_calls += 1
        return self._base.step(action)

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
    def unwrapped(self):
        return self._base.unwrapped

    @property
    def is_batched(self):
        return self._base.is_batched

    @property
    def num_envs(self):
        return self._base.num_envs


class _FakePeriodicEvalEnv:
    def __init__(self, *, num_envs: int = 3, episode_len: int = 1) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.action_widths: list[int] = []
        self._t = 0

    def reset(self, *, seed: int | None = None):
        del seed
        self._t = 0
        return np.zeros((self.num_envs, 3), dtype=np.float32)

    def step(self, actions):
        self.action_widths.append(int(np.asarray(actions).shape[0]))
        self._t += 1
        rewards = np.ones((self.num_envs,), dtype=np.float32)
        dones = np.full((self.num_envs,), self._t >= self.episode_len)
        return np.zeros((self.num_envs, 3), dtype=np.float32), rewards, dones, [{} for _ in range(self.num_envs)]

    def close(self) -> None:
        return


def test_training_loop_eval_period_none_does_not_touch_factory(cartpole_env: TrainingEnv) -> None:
    def _factory(task):
        raise AssertionError(f"eval factory should not run, got {task!r}")

    config = BaseAgentConfig(learning_starts=10, train_freq=1, batch_size=1, seed=0)
    loop = TrainingLoop(
        env=cartpole_env,
        planner=_RandomPlanner(cartpole_env.action_space),
        learner=_NoOpLearner(),
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=config,
    )
    loop.run(4, eval_period=None, eval_env_factory=_factory)
    assert all("eval/mean_return" not in entry for entry in loop.logger.history)


def test_training_loop_eval_period_does_not_step_train_env(cartpole_env: TrainingEnv) -> None:
    env = _EvalCountingEnv(cartpole_env)
    created: list[object] = []

    def factory(task):
        eval_env = _FakePeriodicEvalEnv()
        created.append(eval_env)
        return eval_env

    config = BaseAgentConfig(learning_starts=10, train_freq=1, batch_size=1, seed=0)
    planner = _RandomPlanner(cartpole_env.action_space)
    loop = TrainingLoop(
        env=env,
        planner=planner,
        learner=_NoOpLearner(),
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=config,
    )
    loop.run(6, eval_period=3, eval_env_factory=factory)
    assert env.step_calls == 6
    assert created
    eval_entries = [entry for entry in loop.logger.history if "eval/mean_return" in entry]
    assert len(eval_entries) == 2
    assert all(width == 3 for env_obj in created for width in env_obj.action_widths)


def test_training_loop_eval_period_batched_runs_after_chunks() -> None:
    num_envs = 2
    env = _MockBatchedEnv(num_envs=num_envs)
    collect_calls = {"n": 0}
    original = env.collect_rollout

    def counting_collect(policy, num_steps, *, key=None, on_step=None):
        collect_calls["n"] += 1
        return original(policy, num_steps, key=key, on_step=on_step)

    env.collect_rollout = counting_collect  # type: ignore[method-assign]
    created: list[_FakePeriodicEvalEnv] = []

    def factory(task):
        eval_env = _FakePeriodicEvalEnv()
        created.append(eval_env)
        return eval_env

    class _TrackingBar:
        def __init__(self) -> None:
            self.pulses: list[dict[str, object]] = []

        def start(self, total: int, **kwargs) -> None:
            del total, kwargs

        def update(self, step_info=None, *, n: int = 1) -> None:
            del step_info, n

        def pulse(self, step_info=None) -> None:
            if step_info is not None:
                self.pulses.append(dict(step_info))

        def close(self) -> None:
            return

    config = BaseAgentConfig(
        learning_starts=0,
        train_freq=1,
        batch_size=1,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=1,
    )
    progress_bar = _TrackingBar()
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=num_envs),
        learner=_NoOpLearner(),
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=config,
    )
    train_collects_before = collect_calls["n"]
    loop.run(8, eval_period=4, eval_env_factory=factory, progress_bar=progress_bar)
    assert collect_calls["n"] > train_collects_before
    eval_entries = [entry for entry in loop.logger.history if "eval/mean_return" in entry]
    assert eval_entries
    assert any(pulse.get("phase") == "eval" for pulse in progress_bar.pulses)
    assert any("eval_step" in pulse for pulse in progress_bar.pulses)
    assert all(width == 3 for eval_env in created for width in eval_env.action_widths)


def test_training_loop_eval_period_runs_once_per_cl_task() -> None:
    class _CWRaw:
        benchmark_name = "CW10"
        num_tasks = 2
        task_names = ("hammer-v3", "push-v3")

    env = _MockBatchedEnv(num_envs=2)
    env.raw = _CWRaw()
    tasks_seen: list[str] = []

    def factory(task):
        tasks_seen.append(task.name)
        return _FakePeriodicEvalEnv()

    config = BaseAgentConfig(
        learning_starts=0,
        train_freq=1,
        batch_size=1,
        seed=0,
        jax_rollout_chunk=2,
        gradient_steps_per_rollout=1,
    )
    loop = TrainingLoop(
        env=env,
        planner=_BatchedPlanner(batch_size=2),
        learner=_NoOpLearner(),
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=config,
    )
    loop.run(4, eval_period=4, eval_env_factory=factory)
    assert tasks_seen == ["hammer-v3", "push-v3"]
    eval_entries = [entry for entry in loop.logger.history if "eval/mean_return" in entry]
    assert eval_entries
    assert "eval/task/hammer-v3/mean_return" in eval_entries[0]
    assert "eval/task/push-v3/mean_return" in eval_entries[0]


def test_training_loop_eval_period_cartpole_infers_gym_env(cartpole_env: TrainingEnv) -> None:
    config = BaseAgentConfig(learning_starts=10, train_freq=1, batch_size=1, seed=0)
    loop = TrainingLoop(
        env=cartpole_env,
        planner=_RandomPlanner(cartpole_env.action_space),
        learner=_NoOpLearner(),
        replay_buffer=UniformReplayBuffer(capacity=100),
        config=config,
    )
    loop.run(4, eval_period=4)
    eval_entries = [entry for entry in loop.logger.history if "eval/mean_return" in entry]
    assert len(eval_entries) == 1
    assert eval_entries[0]["eval/mean_return"] >= 1.0
