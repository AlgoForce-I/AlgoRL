"""JAX planner factory."""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.planner import Planner
from algorl.core.types import Action, Observation


class _StubPlanner:
    """Placeholder planner until search implementations land."""

    def __init__(self, kind: str, backend: Backend) -> None:
        self.kind = kind
        self.backend = backend

    def search(self, observation: Observation, **kwargs: Any) -> Action:
        raise NotImplementedError(f"JAX planner {self.kind!r} is not implemented yet.")


def create(kind: str, backend: Backend, **kwargs: Any) -> Planner:
    return _StubPlanner(kind, backend)
