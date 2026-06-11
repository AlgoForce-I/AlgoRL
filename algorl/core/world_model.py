"""World model protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from algorl.core.types import Action, Observation


@runtime_checkable
class WorldModel(Protocol):
    """Latent dynamics model shared by Dreamer and search-based agents.

    Implementations live in ``backends/<backend>/world_models/``.
    """

    def encode(self, observation: Observation) -> Any:
        """Map a Gymnasium observation to a latent state."""

    def transition(self, latent_state: Any, action: Action) -> Any:
        """Predict the next latent state from the current latent state and action."""

    def reward(self, latent_state: Any, action: Action) -> Any:
        """Predict immediate reward; used during search and training."""

    def value(self, latent_state: Any) -> Any:
        """Predict state value; used during search and for value targets."""
