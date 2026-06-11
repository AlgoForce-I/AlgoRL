"""PyTorch world model factory (stub)."""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.world_model import WorldModel


def create(kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
    raise NotImplementedError(
        "PyTorch world models are not implemented yet. Use backend='jax'."
    )
