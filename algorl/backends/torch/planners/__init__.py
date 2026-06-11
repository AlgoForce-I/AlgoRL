"""PyTorch planner factory (stub)."""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.planner import Planner


def create(kind: str, backend: Backend, **kwargs: Any) -> Planner:
    raise NotImplementedError(
        "PyTorch planners are not implemented yet. Use backend='jax'."
    )
