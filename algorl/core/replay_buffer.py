"""Replay buffer protocol.

Generic buffers live in ``algorl/buffers/``. Algorithm-specific buffers may add
fields such as MCTS policy targets and value prefixes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from algorl.core.types import Batch, Transition


@runtime_checkable
class ReplayBuffer(Protocol):
    """Storage for environment transitions and search data."""

    def add(self, transition: Transition) -> None:
        """Store a transition."""

    def sample(self, batch_size: int) -> Batch:
        """Sample a training batch."""

    def __len__(self) -> int:
        """Return the number of stored transitions."""
