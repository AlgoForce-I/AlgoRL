"""Reanalyze batching tests."""

from __future__ import annotations

import numpy as np

from algorl.backends.jax.learners.efficientzero.reanalyze import reanalyze_training_batch


class _RecordingSearchResult:
    def __init__(self, batch_size: int, *, action_dim: int = 1, num_candidates: int = 2) -> None:
        self.batch_size = batch_size
        self.action_weights = np.full((batch_size, num_candidates), 0.5, dtype=np.float32)
        self.root_values = np.arange(batch_size, dtype=np.float32)
        self.root_candidates = np.zeros((batch_size, num_candidates, action_dim), dtype=np.float32)
        self.actions = np.zeros((batch_size, action_dim), dtype=np.float32)
        self.action_indices = np.zeros((batch_size,), dtype=np.int32)


class _RecordingPlanner:
    def __init__(self, *, search_batch_size: int) -> None:
        self.search_batch_size = search_batch_size
        self.batch_sizes: list[int] = []
        self.search_params: list[object | None] = []
        self.use_self_play_flags: list[bool] = []

    def search_batch(
        self,
        observations,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        params=None,
        use_self_play: bool = True,
        **kwargs,
    ):
        del deterministic, temperature, kwargs
        batch_size = len(observations) if isinstance(observations, list) else int(np.asarray(observations).shape[0])
        self.batch_sizes.append(batch_size)
        self.search_params.append(params)
        self.use_self_play_flags.append(use_self_play)
        return _RecordingSearchResult(batch_size)


_REANALYZE_PARAMS = {"model": "reanalyze"}


def test_normalize_observation_batch_accepts_numpy_stack() -> None:
    from algorl.backends.jax.planners.mcts.core import normalize_observation_batch

    batch = normalize_observation_batch(np.zeros((4, 3), dtype=np.float32))
    assert batch.is_stacked is True
    assert batch.batch_size == 4


def test_reanalyze_uses_planner_search_batch_size() -> None:
    planner = _RecordingPlanner(search_batch_size=4)
    observations = np.zeros((5, 3, 2), dtype=np.float32)
    reanalyze_training_batch(
        planner,
        observations,
        params=_REANALYZE_PARAMS,
        reanalyze_count=5,
    )
    assert planner.batch_sizes == [4, 4, 4, 4]


def test_reanalyze_search_batch_size_override() -> None:
    planner = _RecordingPlanner(search_batch_size=1)
    observations = np.zeros((5, 3, 2), dtype=np.float32)
    from algorl.backends.jax.learners.efficientzero.reanalyze import reanalyze_policy_batch

    reanalyze_policy_batch(
        planner,
        observations,
        params=_REANALYZE_PARAMS,
        reanalyze_count=5,
        search_batch_size=4,
    )
    assert planner.batch_sizes == [4, 4, 4, 4]


def test_reanalyze_search_uses_reanalyze_params() -> None:
    planner = _RecordingPlanner(search_batch_size=8)
    observations = np.zeros((2, 4, 3), dtype=np.float32)
    reanalyze_training_batch(
        planner,
        observations,
        params=_REANALYZE_PARAMS,
        reanalyze_count=2,
    )
    assert planner.search_params
    assert all(params is _REANALYZE_PARAMS for params in planner.search_params)
    assert all(use_self_play is False for use_self_play in planner.use_self_play_flags)


def test_resolve_reanalyze_search_width_passthrough_and_fallback() -> None:
    from unittest import mock

    from algorl.agents.configs import EfficientZeroConfig
    from algorl.backends.jax.learners.efficientzero.reanalyze import (
        resolve_reanalyze_search_width,
    )

    explicit = EfficientZeroConfig(reanalyze_search_batch_size=512)
    assert resolve_reanalyze_search_width(explicit) == 512

    unset = EfficientZeroConfig(reanalyze_search_batch_size=None, search_batch_size=8)
    assert resolve_reanalyze_search_width(unset) == 8

    auto = EfficientZeroConfig(reanalyze_search_batch_size="auto")
    with mock.patch(
        "algorl.backends.jax.memory.gpu_available_memory_bytes",
        return_value=None,
    ):
        assert resolve_reanalyze_search_width(auto) == 10_240

    with mock.patch(
        "algorl.backends.jax.memory.gpu_available_memory_bytes",
        return_value=64 * 1024**3,
    ):
        width = resolve_reanalyze_search_width(auto)
        assert 1_024 <= width <= 65_536

    import pytest

    invalid = EfficientZeroConfig(reanalyze_search_batch_size="huge")
    with pytest.raises(ValueError, match="'auto'"):
        resolve_reanalyze_search_width(invalid)


def test_reanalyze_scatters_values_to_correct_positions() -> None:
    """Each root's search value must land at its (batch, step) slot."""
    from algorl.backends.jax.learners.efficientzero.reanalyze import (
        reanalyze_fused_policy_batches,
    )

    class _CountingPlanner:
        search_batch_size = 4

        def __init__(self) -> None:
            self._offset = 0

        def search_batch(self, observations, **kwargs):
            del kwargs
            batch_size = int(np.asarray(observations).shape[0])
            result = _RecordingSearchResult(batch_size)
            result.root_values = self._offset + np.arange(batch_size, dtype=np.float32)
            self._offset += batch_size
            return result

    window = 3
    outputs = reanalyze_fused_policy_batches(
        _CountingPlanner(),  # type: ignore[arg-type]
        [np.zeros((4, window, 2), dtype=np.float32), np.zeros((4, window, 2), dtype=np.float32)],
        params={},
        reanalyze_count=2,
        search_batch_size=4,
    )

    flat = 0
    for policy_targets, search_values, _, _ in outputs:
        assert policy_targets.shape[0] == 4
        for batch_index in range(2):  # reanalyze_count
            for step_index in range(window):
                assert search_values[batch_index, step_index] == flat
                flat += 1
        # Rows beyond reanalyze_count stay zeroed.
        assert np.all(search_values[2:] == 0.0)


def test_reanalyze_preserves_trajectory_shapes() -> None:
    planner = _RecordingPlanner(search_batch_size=8)
    observations = np.zeros((2, 4, 3), dtype=np.float32)
    policy_targets, search_values, policy_candidates, best_actions = reanalyze_training_batch(
        planner,
        observations,
        params=_REANALYZE_PARAMS,
        reanalyze_count=2,
    )
    assert policy_targets.shape == (2, 4, 2)
    assert search_values.shape == (2, 4)
    assert policy_candidates.shape == (2, 4, 2, 1)
    assert best_actions.shape == (2, 4, 1)
