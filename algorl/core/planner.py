"""Planner protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from algorl.core.types import Action, Observation


@runtime_checkable
class Planner(Protocol):
    """Reasoning mechanism used at decision time.

    Implementations live in ``backends/<backend>/planners/``.
    """

    def search(self, observation: Observation, **kwargs: Any) -> Action:
        """Return an action for ``observation`` after planning or imagination."""
