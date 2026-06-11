"""PyTorch learner factory (stub).

When the PyTorch backend is added, mirror ``backends/jax/learners/``:
- ``efficient_zero.py``, ``muzero.py``, ``dreamer.py``, etc.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.learner import Learner


def create(kind: str, backend: Backend, **kwargs: Any) -> Learner:
    # Implement: PyTorch versions of the learners in backends/jax/learners/.
    raise NotImplementedError(
        "PyTorch learners are not implemented yet. Use backend='jax'."
    )
