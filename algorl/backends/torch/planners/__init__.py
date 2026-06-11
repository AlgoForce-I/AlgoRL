"""PyTorch planner factory (stub).

When the PyTorch backend is added, mirror ``backends/jax/planners/``:
- ``mcts.py``, ``imagination.py``, ``cem.py``, ``mpc.py``
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.planner import Planner


def create(kind: str, backend: Backend, **kwargs: Any) -> Planner:
    # Implement: PyTorch versions of the planners in backends/jax/planners/.
    raise NotImplementedError(
        "PyTorch planners are not implemented yet. Use backend='jax'."
    )
