"""PyTorch world model factory (stub).

When the PyTorch backend is added, mirror the JAX layout:
- ``efficient_zero.py``, ``rssm.py``, ``td_mpc.py``, etc.
- register each kind in ``create()`` the same way as ``backends/jax/world_models/``.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.world_model import WorldModel


def create(kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
    # Implement: PyTorch versions of the world models in backends/jax/world_models/.
    raise NotImplementedError(
        "PyTorch world models are not implemented yet. Use backend='jax'."
    )
