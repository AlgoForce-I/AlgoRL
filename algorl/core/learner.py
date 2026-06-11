"""Learner abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from algorl.core.replay_buffer import ReplayBuffer


class Learner(ABC):
    """Training loop responsible for parameter updates.

    Implementations must subclass this class in ``backends/<backend>/learners/``.
    """

    @abstractmethod
    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        """Sample from ``replay_buffer``, update parameters, and return metric scalars."""
