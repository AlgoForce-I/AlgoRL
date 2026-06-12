"""Planner hierarchy tests."""

from __future__ import annotations

from typing import Any

import mctx
import pytest

from algorl.backends.jax.planners.mcts.core import BaseMCTSPlanner, NormalizedObservationBatch
from algorl.core.component_context import ComponentContext
from algorl.core.planner import BatchedPlanner, Planner


class _ConcreteMCTSPlanner(BaseMCTSPlanner):
    def build_root(self, observations: NormalizedObservationBatch) -> mctx.RootFnOutput:
        raise NotImplementedError

    def make_recurrent_fn(self, observations: NormalizedObservationBatch) -> mctx.RecurrentFn:
        raise NotImplementedError


@pytest.fixture
def mcts_planner(cartpole_env) -> _ConcreteMCTSPlanner:
    from algorl.agents.configs import AlphaZeroConfig
    from algorl.backends.jax.backend import JAXBackend

    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=cartpole_env,
    )
    return _ConcreteMCTSPlanner(context)


def test_base_mcts_planner_inherits_batched_planner() -> None:
    assert issubclass(BaseMCTSPlanner, BatchedPlanner)
    assert issubclass(BaseMCTSPlanner, Planner)
    assert issubclass(BatchedPlanner, Planner)


def test_mcts_planner_instance_types(mcts_planner: _ConcreteMCTSPlanner) -> None:
    assert isinstance(mcts_planner, Planner)
    assert isinstance(mcts_planner, BatchedPlanner)


def test_batched_planner_declares_search_batch() -> None:
    assert getattr(BatchedPlanner.search_batch, "__isabstractmethod__", False)


def test_search_batch_requires_initialized_params(mcts_planner: _ConcreteMCTSPlanner) -> None:
    with pytest.raises(RuntimeError, match="params are not initialized"):
        mcts_planner.search_batch([0])


def test_iter_search_chunks_uses_config_batch_size(cartpole_env) -> None:
    from algorl.agents.configs import AlphaZeroConfig
    from algorl.backends.jax.backend import JAXBackend

    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False, search_batch_size=4),
        env=cartpole_env,
    )
    planner = _ConcreteMCTSPlanner(context)
    chunks = list(planner.iter_search_chunks([1, 2, 3, 4, 5]))
    assert chunks == [[1, 2, 3, 4], [5]]


def test_resolve_mcts_config_preserves_stochastic_settings() -> None:
    from algorl.backends.jax.planners.mcts.core import MCTSConfig

    planner = _ConcreteMCTSPlanner.__new__(_ConcreteMCTSPlanner)
    planner.mcts_config = MCTSConfig(
        num_simulations=8,
        temperature=1.0,
        dirichlet_fraction=0.25,
        use_gumbel=False,
    )
    resolved = planner._resolve_mcts_config(None, deterministic=False)
    assert resolved.temperature == 1.0
    assert resolved.dirichlet_fraction == 0.25


def test_search_batch_size_rejects_invalid_config(cartpole_env) -> None:
    from algorl.agents.configs import AlphaZeroConfig
    from algorl.backends.jax.backend import JAXBackend

    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False, search_batch_size=0),
        env=cartpole_env,
    )
    planner = _ConcreteMCTSPlanner(context)
    with pytest.raises(ValueError, match="search_batch_size must be > 0"):
        _ = planner.search_batch_size


def test_resolve_mcts_config_applies_deterministic_to_override() -> None:
    from algorl.backends.jax.planners.mcts.core import MCTSConfig

    planner = _ConcreteMCTSPlanner.__new__(_ConcreteMCTSPlanner)
    planner.mcts_config = MCTSConfig(
        num_simulations=8,
        temperature=1.0,
        dirichlet_fraction=0.25,
        use_gumbel=True,
        gumbel_scale=1.0,
    )
    override = MCTSConfig(
        num_simulations=16,
        temperature=1.0,
        dirichlet_fraction=0.25,
        use_gumbel=True,
        gumbel_scale=1.0,
    )
    resolved = planner._resolve_mcts_config(override, deterministic=True)
    assert resolved.temperature == 0.0
    assert resolved.dirichlet_fraction == 0.0
    assert resolved.gumbel_scale == 0.0
