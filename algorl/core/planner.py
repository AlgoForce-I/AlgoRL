"""Planner abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from algorl.core.types import Action, Observation


class Planner(ABC):
    """Reasoning mechanism used at decision time.

    Implementations must subclass this class in ``backends/<backend>/planners/``.
    """

    @abstractmethod
    def search(self, observation: Observation, **kwargs: Any) -> Action:
        """Return an action for ``observation`` after planning or imagination."""
