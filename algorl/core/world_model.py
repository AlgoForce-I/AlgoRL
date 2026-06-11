"""World model abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from algorl.core.types import Action, Observation


class WorldModel(ABC):
    """Latent dynamics model shared by Dreamer and search-based agents.

    Implementations must subclass this class in ``backends/<backend>/world_models/``.
    """

    @abstractmethod
    def encode(self, observation: Observation) -> Any:
        """Map a Gymnasium observation to a latent state."""

    @abstractmethod
    def transition(self, latent_state: Any, action: Action) -> Any:
        """Predict the next latent state from the current latent state and action."""

    @abstractmethod
    def reward(self, latent_state: Any, action: Action) -> Any:
        """Predict immediate reward; used during search and training."""

    @abstractmethod
    def value(self, latent_state: Any) -> Any:
        """Predict state value; used during search and for value targets."""
