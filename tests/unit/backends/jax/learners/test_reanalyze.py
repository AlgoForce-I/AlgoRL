"""Reanalyze batching tests."""

from __future__ import annotations

import numpy as np

from algorl.backends.jax.learners.efficientzero.reanalyze import reanalyze_training_batch


class _RecordingSearchResult:
    def __init__(self, batch_size: int, *, action_dim: int = 1, num_candidates: int = 2) -> None:
        self.batch_size = batch_size
        self._action_dim = action_dim
        self._num_candidates = num_candidates

    def to_single(self, index: int) -> object:
        return _RecordingSingleResult(
            index=index,
            action_dim=self._action_dim,
            num_candidates=self._num_candidates,
        )


class _RecordingSingleResult:
    def __init__(self, *, index: int, action_dim: int, num_candidates: int) -> None:
        self.action_weights = np.full((num_candidates,), 0.5, dtype=np.float32)
        self.root_value = float(index)
        self.root_candidates = np.zeros((num_candidates, action_dim), dtype=np.float32)
        self.action = np.zeros((action_dim,), dtype=np.float32)


class _RecordingPlanner:
    def __init__(self, *, search_batch_size: int) -> None:
        self.search_batch_size = search_batch_size
        self.batch_sizes: list[int] = []

    def search_batch(self, observations, *, deterministic: bool = False):
        del deterministic
        batch_size = len(observations) if isinstance(observations, list) else int(np.asarray(observations).shape[0])
        self.batch_sizes.append(batch_size)
        return _RecordingSearchResult(batch_size)


def test_normalize_observation_batch_accepts_numpy_stack() -> None:
    from algorl.backends.jax.planners.mcts.core import normalize_observation_batch

    batch = normalize_observation_batch(np.zeros((4, 3), dtype=np.float32))
    assert batch.is_stacked is True
    assert batch.batch_size == 4


def test_reanalyze_uses_planner_search_batch_size() -> None:
    planner = _RecordingPlanner(search_batch_size=4)
    observations = np.zeros((5, 3, 2), dtype=np.float32)
    reanalyze_training_batch(planner, observations, reanalyze_count=5)
    assert planner.batch_sizes == [4, 4, 4, 3]


def test_reanalyze_preserves_trajectory_shapes() -> None:
    planner = _RecordingPlanner(search_batch_size=8)
    observations = np.zeros((2, 4, 3), dtype=np.float32)
    policy_targets, search_values, policy_candidates, best_actions = reanalyze_training_batch(
        planner,
        observations,
        reanalyze_count=2,
    )
    assert policy_targets.shape == (2, 4, 2)
    assert search_values.shape == (2, 4)
    assert policy_candidates.shape == (2, 4, 2, 1)
    assert best_actions.shape == (2, 4, 1)
