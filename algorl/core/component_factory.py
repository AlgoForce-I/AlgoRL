"""Backend component factory abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from algorl.core.backend import Backend
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.world_model import WorldModel


class ComponentFactory(ABC):
    """Creates algorithm components for one backend implementation."""

    @abstractmethod
    def create_world_model(self, kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
        """Return a world model implementation for ``kind``."""

    @abstractmethod
    def create_planner(self, kind: str, backend: Backend, **kwargs: Any) -> Planner:
        """Return a planner implementation for ``kind``."""

    @abstractmethod
    def create_learner(self, kind: str, backend: Backend, **kwargs: Any) -> Learner:
        """Return a learner implementation for ``kind``."""
