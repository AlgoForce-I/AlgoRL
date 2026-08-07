"""Tests for fused EfficientZero reanalyze."""

from __future__ import annotations

import numpy as np

from algorl.backends.jax.learners.efficientzero.reanalyze import reanalyze_fused_policy_batches
from algorl.backends.jax.planners.efficientzero import EfficientZeroBatchedResult


class _StubPlanner:
    search_batch_size = 4

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def search_batch(self, observations, **kwargs):
        del kwargs
        batch_size = int(np.asarray(observations).shape[0])
        self.call_sizes.append(batch_size)
        return EfficientZeroBatchedResult(
            actions=np.zeros((batch_size, 1), dtype=np.float32),
            action_indices=np.zeros((batch_size,), dtype=np.int32),
            action_weights=np.ones((batch_size, 2), dtype=np.float32),
            root_values=np.zeros((batch_size,), dtype=np.float32),
            root_candidates=np.zeros((batch_size, 2, 1), dtype=np.float32),
        )


def test_reanalyze_fused_policy_batches_queues_all_burst_items() -> None:
    planner = _StubPlanner()
    observations = [
        np.zeros((2, 3, 4), dtype=np.float32),
        np.zeros((2, 3, 4), dtype=np.float32),
    ]
    progress: list[tuple[int, int]] = []

    outputs = reanalyze_fused_policy_batches(
        planner,  # type: ignore[arg-type]
        observations,
        params={},
        reanalyze_count=2,
        search_batch_size=4,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert len(outputs) == 2
    assert sum(planner.call_sizes) == 12
    assert progress[-1] == (12, 12)
    for policy_targets, search_values, policy_candidates, best_actions in outputs:
        assert policy_targets.shape[:2] == (2, 3)
        assert search_values.shape == (2, 3)
        assert policy_candidates.shape[:2] == (2, 3)
        assert best_actions.shape[:2] == (2, 3)
