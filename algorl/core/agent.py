"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from algorl.core.types import Action, Observation


class BaseAgent(ABC):
    """Common interface for all AlgoRL agents."""

    @abstractmethod
    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        """Train the agent for ``total_timesteps`` environment steps."""

    @abstractmethod
    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        """Select an action for ``observation``."""

    def save(self, path: str) -> None:
        """Persist agent state to ``path``."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Restore agent state from ``path``."""
        raise NotImplementedError
