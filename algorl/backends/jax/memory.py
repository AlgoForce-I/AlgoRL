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
