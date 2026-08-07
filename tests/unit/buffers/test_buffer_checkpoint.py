"""Round-trip tests for replay buffer directory checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.buffers.efficientzero import (
    BEST_ACTION_INFO_KEY,
    POLICY_TARGET_INFO_KEY,
    ROOT_CANDIDATES_INFO_KEY,
    SEARCH_VALUE_INFO_KEY,
    EfficientZeroReplayBuffer,
)
from algorl.buffers.replay import UniformReplayBuffer
from algorl.core.types import Transition


def _ez_transition(index: int, *, done: bool = False, env_id: int = 0) -> Transition:
    policy = np.full((4,), 0.25, dtype=np.float32)
    candidates = np.stack(
        [np.asarray([0.1], dtype=np.float32) for _ in range(4)],
        axis=0,
    )
    return Transition(
        observation=np.full((4,), float(index), dtype=np.float32),
        action=np.asarray([0.1], dtype=np.float32),
        reward=float(index) * 0.1,
        next_observation=np.full((4,), float(index + 1), dtype=np.float32),
        done=done,
        info={
            POLICY_TARGET_INFO_KEY: policy,
            SEARCH_VALUE_INFO_KEY: float(index) * 0.05,
            ROOT_CANDIDATES_INFO_KEY: candidates,
            BEST_ACTION_INFO_KEY: np.asarray([0.1], dtype=np.float32),
            "env_id": env_id,
        },
    )


def test_efficient_zero_buffer_roundtrip(tmp_path: Path) -> None:
    config = EfficientZeroConfig(
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        use_priority=True,
        reanalyze_ratio=0.0,
    )
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    for index in range(12):
        buffer.add(_ez_transition(index, done=(index % 4 == 3)))

    assert len(buffer) > 0
    lookup_before = list(buffer._lookup)
    priorities_before = list(buffer._priorities)
    base_before = buffer._base_traj_idx
    commits_before = buffer._total_commits
    n_traj_before = len(buffer._stored_steps)

    buffer.save(tmp_path / "ez_buf")

    restored = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    restored.load(tmp_path / "ez_buf")

    assert len(restored) == len(buffer)
    assert restored._base_traj_idx == base_before
    assert restored._total_commits == commits_before
    assert len(restored._stored_steps) == n_traj_before
    assert restored._lookup == lookup_before
    np.testing.assert_allclose(restored._priorities, priorities_before)

    # Sampling still works and observations match stored content.
    batch = restored.sample(2, beta=1.0, trained_steps=0)
    assert "observations" in batch.data
    assert np.asarray(batch.data["observations"]).shape[0] == 2


def test_efficient_zero_buffer_preserves_inflight_lanes(tmp_path: Path) -> None:
    config = EfficientZeroConfig(
        batch_size=2,
        unroll_steps=2,
        trajectory_size=8,
        use_priority=False,
        reanalyze_ratio=0.0,
    )
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=2,
        trajectory_size=8,
    )
    # Not enough steps to commit a full block — stays in _active.
    for index in range(3):
        buffer.add(_ez_transition(index, done=False, env_id=0))
    assert len(buffer._active) == 1
    assert len(buffer) == 0

    buffer.save(tmp_path / "inflight")
    restored = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=2,
        trajectory_size=8,
    )
    restored.load(tmp_path / "inflight")
    assert len(restored._active) == 1
    assert len(restored._active[0].steps) == 3
    np.testing.assert_array_equal(
        restored._active[0].steps[0].observation,
        buffer._active[0].steps[0].observation,
    )


def test_efficient_zero_buffer_capacity_mismatch(tmp_path: Path) -> None:
    config = EfficientZeroConfig(unroll_steps=2, trajectory_size=4)
    buffer = EfficientZeroReplayBuffer(capacity=50, config=config, unroll_steps=2, trajectory_size=4)
    buffer.save(tmp_path / "buf")
    other = EfficientZeroReplayBuffer(capacity=51, config=config, unroll_steps=2, trajectory_size=4)
    with pytest.raises(ValueError, match="capacity"):
        other.load(tmp_path / "buf")


def test_uniform_buffer_roundtrip(tmp_path: Path) -> None:
    buffer = UniformReplayBuffer(capacity=10)
    for index in range(5):
        buffer.add(
            Transition(
                observation=np.asarray([index], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                reward=float(index),
                next_observation=np.asarray([index + 1], dtype=np.float32),
                done=False,
                info={"tag": index},
            )
        )
    buffer.save(tmp_path / "uniform")
    restored = UniformReplayBuffer(capacity=10)
    restored.load(tmp_path / "uniform")
    assert len(restored) == 5
    assert float(restored._storage[2].reward) == 2.0
    assert restored._storage[2].info["tag"] == 2
