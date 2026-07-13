"""JAX GPU memory helpers.

Call :func:`configure_jax_gpu_memory` before importing JAX when possible
(``example.py`` does this at module top). It also pins spawned Gymnasium
vector-env workers to CPU JAX so re-importing user scripts does not claim GPU
memory in every subprocess.
"""

from __future__ import annotations

import multiprocessing as mp
import os


def _configure_jax_for_subprocess_workers() -> None:
    if mp.current_process().name == "MainProcess":
        return
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def gpu_available_memory_bytes() -> int | None:
    """Free device memory on the default JAX accelerator, or ``None`` if unknown.

    Uses the allocator's own view (``bytes_limit`` respects
    ``XLA_PYTHON_CLIENT_MEM_FRACTION``), so results account for memory already
    claimed by compiled programs and live buffers.
    """
    try:
        import jax

        device = jax.local_devices()[0]
        if device.platform == "cpu":
            return None
        stats = device.memory_stats()
        if not stats:
            return None
        limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
        if limit is None:
            return None
        in_use = stats.get("bytes_in_use", 0)
        return max(0, int(limit) - int(in_use))
    except Exception:  # pragma: no cover - platform-specific probing
        return None


def configure_jax_gpu_memory(
    *,
    preallocate: bool = False,
    memory_fraction: float | None = 0.85,
) -> None:
    """Tune XLA GPU allocator for long EZ runs (EfficientZero-V2-like PyTorch footprint).

    JAX defaults to preallocating nearly all GPU memory, which makes OOM look
    sudden and leaves little room for MJX + MCTS + backward peaks.

    When Gymnasium ``AsyncVectorEnv`` uses ``spawn``, worker processes re-import
    the user script. Call this function before any JAX import so workers only
    initialize CPU JAX and avoid GPU OOM during env startup.
    """
    _configure_jax_for_subprocess_workers()
    if not preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if memory_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(memory_fraction))
