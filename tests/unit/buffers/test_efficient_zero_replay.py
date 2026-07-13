"""EfficientZero replay buffer tests."""

import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.buffers.efficientzero import (
    ENV_ID_INFO_KEY,
    POLICY_TARGET_INFO_KEY,
    PRED_VALUE_INFO_KEY,
    SEARCH_VALUE_INFO_KEY,
    EfficientZeroReplayBuffer,
    step_from_transition,
)
from algorl.core.types import Transition


def _transition(
    index: int,
    *,
    done: bool = False,
    policy_dim: int = 4,
    env_id: int | None = None,
) -> Transition:
    policy = np.full((policy_dim,), 1.0 / policy_dim, dtype=np.float32)
    info = {
        POLICY_TARGET_INFO_KEY: policy,
        SEARCH_VALUE_INFO_KEY: float(index) * 0.1,
        PRED_VALUE_INFO_KEY: float(index) * 0.05,
    }
    if env_id is not None:
        info[ENV_ID_INFO_KEY] = env_id
    return Transition(
      observation=np.full((3,), float(index), dtype=np.float32),
      action=np.asarray([0.25], dtype=np.float32),
      reward=float(index),
      next_observation=np.full((3,), float(index + 1), dtype=np.float32),
      done=done,
      info=info,
  )


def test_step_from_transition_extracts_search_targets() -> None:
    transition = _transition(2, policy_dim=3)
    step = step_from_transition(transition)

    assert step.observation.shape == (3,)
    assert step.action.shape == (1,)
    assert step.reward == 2.0
    assert step.policy_target.shape == (3,)
    assert step.search_value == pytest.approx(0.2)
    assert step.pred_value == pytest.approx(0.1)
    assert step.done is False


def test_efficient_zero_buffer_chunks_trajectories() -> None:
    config = EfficientZeroConfig(unroll_steps=2, trajectory_size=4, use_priority=False)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=2,
        trajectory_size=4,
    )
    for index in range(4):
        buffer.add(_transition(index, done=index == 3))

    # Every core transition is a valid sample position.
    assert len(buffer) == 4

    # Observations/rewards expose the extended target window
    # (unroll + 1 + td_steps for bootstrapped value targets).
    ext_window = 2 + 1 + config.td_steps
    batch = buffer.sample(2)
    assert batch.data["observations"].shape == (2, ext_window, 3)
    assert batch.data["actions"].shape == (2, 2, 1)
    assert batch.data["rewards"].shape == (2, ext_window)
    assert batch.data["value_targets"].shape == (2, 3)
    assert batch.data["weights"].shape == (2,)
    assert batch.data["policy_candidates"].shape[0] == 2
    assert batch.data["valid_lengths"].shape == (2,)
    assert batch.data["bootstrap_limits"].shape == (2,)


def test_efficient_zero_buffer_masks_and_lengths_near_trajectory_end() -> None:
    config = EfficientZeroConfig(unroll_steps=2, trajectory_size=4, use_priority=False)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=2,
        trajectory_size=4,
    )
    for index in range(4):
        buffer.add(_transition(index, done=index == 3))

    batch = buffer._build_batch(np.asarray([0, 2, 3]), trained_steps=0)
    # valid length = core_len - position.
    assert batch.data["valid_lengths"].tolist() == [4, 2, 1]
    # Terminal observation is stored, so bootstrap may peek one step past the end.
    assert batch.data["bootstrap_limits"].tolist() == [4, 2, 1]
    # Unroll mask: step k trains only when position k+1 is in trajectory.
    assert batch.data["masks"].tolist() == [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]


def test_efficient_zero_buffer_commits_partial_trajectory_on_done() -> None:
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        unroll_steps=2,
        trajectory_size=10,
    )
    for index in range(3):
        buffer.add(_transition(index, done=index == 2))

    assert len(buffer) == 3


def test_efficient_zero_buffer_respects_capacity() -> None:
    buffer = EfficientZeroReplayBuffer(
        capacity=3,
        unroll_steps=1,
        trajectory_size=2,
    )
    for index in range(6):
        buffer.add(_transition(index, done=index in {1, 3, 5}))

    assert len(buffer) >= 2


def test_new_transitions_get_optimistic_max_priority() -> None:
    """Fresh transitions inherit buffer-wide max priority."""
    config = EfficientZeroConfig(unroll_steps=1, trajectory_size=2, use_priority=True)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=1,
        trajectory_size=2,
    )
    for index in range(4):
        buffer.add(_transition(index, done=index in {1, 3}))

    priorities = buffer._priorities
    assert len(priorities) >= 2
    # max(buffer_max, trajectory_max) never decreases across stores.
    assert all(a <= b for a, b in zip(priorities, priorities[1:]))
    assert all(p >= 1.0 for p in priorities)

    # Simulate a training refresh raising one priority; the next trajectory
    # must inherit the new buffer-wide max.
    buffer.update_priorities(np.asarray([0]), np.asarray([42.0]))
    for index in range(4, 6):
        buffer.add(_transition(index, done=index == 5))
    assert buffer._priorities[-1] == pytest.approx(42.0)


def test_efficient_zero_buffer_keeps_interleaved_env_streams_separate() -> None:
    """Step-major adds from parallel envs must not mix trajectories."""
    config = EfficientZeroConfig(unroll_steps=2, trajectory_size=4, use_priority=False)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=2,
        trajectory_size=4,
    )

    num_steps = 8
    # Interleave two lanes exactly like the batched rollout does:
    # (step 0, env 0), (step 0, env 1), (step 1, env 0), ...
    for step in range(num_steps):
        for env_id, base in ((0, 0), (1, 100)):
            buffer.add(
                _transition(
                    base + step,
                    done=step == num_steps - 1,
                    env_id=env_id,
                )
            )

    assert len(buffer) == 2 * num_steps
    assert not buffer._pending_commit
    for steps in buffer._stored_steps:
        observed = [float(step.observation[0]) for step in steps]
        # Each stored trajectory belongs to a single env lane...
        assert all(obs < 100.0 for obs in observed) or all(obs >= 100.0 for obs in observed)
        # ... and is in temporal order with consecutive steps.
        assert observed == sorted(observed)
        assert all(b - a == 1.0 for a, b in zip(observed, observed[1:]))


def test_efficient_zero_buffer_releases_pending_block_after_gap_tail() -> None:
    """A full block becomes sampleable once the next block has its tail context.

    Waiting for the next block to *fill* delays data availability by a whole
    trajectory block (trajectory_size * num_envs env steps in batched runs).
    """
    config = EfficientZeroConfig(
        unroll_steps=2,
        trajectory_size=10,
        td_steps=2,
        use_priority=False,
    )
    buffer = EfficientZeroReplayBuffer(
        capacity=1000,
        config=config,
        unroll_steps=2,
        trajectory_size=10,
    )
    gap = 1 + 2  # n_stack + td_steps for bootstrapped targets

    for index in range(10):
        buffer.add(_transition(index))
    # Block is full but has no tail context yet: held back.
    assert len(buffer) == 0
    assert 0 in buffer._pending_commit

    for index in range(10, 10 + gap - 1):
        buffer.add(_transition(index))
    assert len(buffer) == 0

    buffer.add(_transition(10 + gap - 1))
    # Gap tail available: pending block stored without waiting for block 2.
    assert len(buffer) == 10
    assert 0 not in buffer._pending_commit
    stored = buffer._stored_steps[0]
    assert len(stored) == 10 + gap
    assert [float(step.observation[0]) for step in stored] == [float(i) for i in range(10 + gap)]


def test_efficient_zero_buffer_default_env_id_matches_sequential() -> None:
    """Transitions without env_id (sequential path) all land in lane 0."""
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        unroll_steps=2,
        trajectory_size=4,
    )
    for index in range(4):
        buffer.add(_transition(index, done=index == 3))
    assert len(buffer) == 4
    assert list(buffer._active.keys()) == [0]


def test_efficient_zero_buffer_rejects_oversized_batch() -> None:
    buffer = EfficientZeroReplayBuffer(
        capacity=10,
        unroll_steps=1,
        trajectory_size=2,
    )
    buffer.add(_transition(0))
    buffer.add(_transition(1))

    with pytest.raises(ValueError, match="exceeds stored transitions"):
        buffer.sample(2)
