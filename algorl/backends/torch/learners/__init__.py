"""PyTorch learner factory (stub)."""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.learner import Learner


def create(kind: str, backend: Backend, **kwargs: Any) -> Learner:
    raise NotImplementedError(
        "PyTorch learners are not implemented yet. Use backend='jax'."
    )
