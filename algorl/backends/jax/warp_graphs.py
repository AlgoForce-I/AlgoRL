"""Keep MJX's warp CUDA-graph cache inside the driver's live-graph limit.

MJX's warp backend captures every physics step into a CUDA graph
(``GraphMode.WARP`` is the default of ``mjx.put_model``) and replays it while the
XLA buffer addresses match, and ``mujoco_warp`` allocates its collision
workspace *inside* that capture. A process can hold only about 2048 live
captured graphs that contain an allocation node; past that, every capture-time
allocation fails with warp's ``Failed to allocate N bytes on device 'cuda:0'``
even with tens of GB of VRAM free.

MJX registers a fresh FFI callable per trace, each with its own graph cache
(32 entries), and the registry holding them is never pruned, so the live graph
count only grows: on CW10 every task switch adds roughly a dozen callables,
i.e. a few hundred more graph slots. Long continual-learning runs therefore hit
the limit mid-rollout after hours of training. Call :func:`enforce_graph_budget`
periodically and :func:`release_graph_caches` when envs are retired; dropped
graphs are re-captured on demand.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

# Stay well under the driver's ~2048 live-graph limit: a rollout chunk or a
# single eval task can still add a few hundred captures before the next check.
DEFAULT_GRAPH_BUDGET = 768

_FFI_MODULE_NAME = "mujoco.mjx.third_party.warp._src.jax_experimental.ffi"


def _ffi_module(*, load: bool = False) -> Any | None:
    """MJX's vendored warp FFI module, or ``None`` when it is not in use."""
    module = sys.modules.get(_FFI_MODULE_NAME)
    if module is not None or not load:
        return module
    try:
        return importlib.import_module(_FFI_MODULE_NAME)
    except ImportError:  # pragma: no cover - MJX built without the warp backend
        return None


def _ffi_callables(module: Any) -> list[Any]:
    registry = getattr(module, "_FFI_CALLABLE_REGISTRY", None)
    if registry is None:  # pragma: no cover - upstream layout change
        return []
    return list(registry.values())


def live_graph_count() -> int | None:
    """Captured CUDA graphs MJX is holding, or ``None`` if MJX-warp is unused."""
    module = _ffi_module()
    if module is None:
        return None
    return sum(len(callable_.captures) for callable_ in _ffi_callables(module))


def release_graph_caches() -> int:
    """Drop every cached graph, freeing its driver-side slot.

    Returns the number of graphs released.
    """
    module = _ffi_module()
    if module is None:
        return 0
    callables = _ffi_callables(module)
    released = sum(len(callable_.captures) for callable_ in callables)
    if released == 0:
        return 0
    clear = getattr(module, "clear_jax_callable_graph_cache", None)
    if clear is not None:
        clear()
    else:  # pragma: no cover - upstream layout change
        for callable_ in callables:
            callable_.captures.clear()
    return released


def enforce_graph_budget(limit: int = DEFAULT_GRAPH_BUDGET) -> int:
    """Release cached graphs once more than ``limit`` of them are live."""
    count = live_graph_count()
    if count is None or count <= limit:
        return 0
    return release_graph_caches()
