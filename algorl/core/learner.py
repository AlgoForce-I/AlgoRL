"""Learner protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from algorl.core.replay_buffer import ReplayBuffer


@runtime_checkable
class Learner(Protocol):
    """Training loop responsible for parameter updates."""

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        """Run one optimization step and return scalar metrics."""
