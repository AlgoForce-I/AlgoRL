"""Replay buffer abstract base class.

Generic buffers live in ``algorl/buffers/``. Algorithm-specific buffers may add
fields such as MCTS policy targets and value prefixes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from algorl.core.types import Batch, Transition


class ReplayBuffer(ABC):
    """Storage for environment transitions and search data.

    Implementations must subclass this class in ``algorl/buffers/``.
    """

    @abstractmethod
    def add(self, transition: Transition) -> None:
        """Store a transition."""

    @abstractmethod
    def sample(self, batch_size: int) -> Batch:
        """Sample a training batch."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored transitions."""
