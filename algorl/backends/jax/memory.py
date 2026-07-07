"""JAX GPU memory helpers.

Call :func:`configure_jax_gpu_memory` before importing JAX when possible
(``example.py`` does this at module top).
"""

from __future__ import annotations

import os


def configure_jax_gpu_memory(
    *,
    preallocate: bool = False,
    memory_fraction: float | None = 0.85,
) -> None:
    """Tune XLA GPU allocator for long EZ runs (EfficientZero-V2-like PyTorch footprint).

    JAX defaults to preallocating nearly all GPU memory, which makes OOM look
    sudden and leaves little room for MJX + MCTS + backward peaks.
    """
    if not preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if memory_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(memory_fraction))
