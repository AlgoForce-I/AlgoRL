"""Tests for the MJX/warp captured-CUDA-graph guard."""

from __future__ import annotations

import collections
import contextlib
from collections.abc import Iterator
from typing import Any
from unittest import mock

from algorl.backends.jax import warp_graphs


class _FakeFfiCallable:
    def __init__(self, graphs: int) -> None:
        self.captures = collections.OrderedDict((index, object()) for index in range(graphs))


class _FakeFfiModule:
    """Stand-in for MJX's vendored ``warp.jax_experimental.ffi`` module."""

    def __init__(self, *graphs_per_callable: int) -> None:
        self._FFI_CALLABLE_REGISTRY = {
            index: _FakeFfiCallable(graphs) for index, graphs in enumerate(graphs_per_callable)
        }
        self.clear_calls = 0

    def clear_jax_callable_graph_cache(self) -> None:
        self.clear_calls += 1
        for callable_ in self._FFI_CALLABLE_REGISTRY.values():
            callable_.captures.clear()


def _installed(module: Any) -> Any:
    return mock.patch.dict("sys.modules", {warp_graphs._FFI_MODULE_NAME: module})


@contextlib.contextmanager
def _without_mjx() -> Iterator[None]:
    with mock.patch.dict("sys.modules", {}, clear=False) as modules:
        modules.pop(warp_graphs._FFI_MODULE_NAME, None)
        yield


def test_live_graph_count_is_none_without_mjx_warp() -> None:
    with _without_mjx():
        assert warp_graphs.live_graph_count() is None


def test_live_graph_count_sums_every_callable_cache() -> None:
    with _installed(_FakeFfiModule(3, 5)):
        assert warp_graphs.live_graph_count() == 8


def test_enforce_graph_budget_leaves_caches_under_the_limit() -> None:
    module = _FakeFfiModule(4, 6)
    with _installed(module):
        assert warp_graphs.enforce_graph_budget(limit=10) == 0
        assert warp_graphs.live_graph_count() == 10
    assert module.clear_calls == 0


def test_enforce_graph_budget_releases_caches_over_the_limit() -> None:
    module = _FakeFfiModule(10, 15)
    with _installed(module):
        assert warp_graphs.enforce_graph_budget(limit=20) == 25
        assert warp_graphs.live_graph_count() == 0
    assert module.clear_calls == 1


def test_release_graph_caches_reports_released_graphs() -> None:
    module = _FakeFfiModule(2, 2)
    with _installed(module):
        assert warp_graphs.release_graph_caches() == 4
        assert warp_graphs.release_graph_caches() == 0
    assert module.clear_calls == 1


def test_release_graph_caches_without_mjx_warp_is_a_noop() -> None:
    with _without_mjx():
        assert warp_graphs.release_graph_caches() == 0


def test_graph_budget_stays_below_the_driver_live_graph_limit() -> None:
    # A rollout chunk or eval task may still add captures before the next check.
    assert warp_graphs.DEFAULT_GRAPH_BUDGET < 2048
