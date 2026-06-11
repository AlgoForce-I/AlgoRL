"""EfficientZero replay buffer (stub).

Implement this buffer before wiring ``EfficientZero.learn()``.
It should store everything the learner needs for one training step.
"""

from __future__ import annotations

from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch, Transition


class EfficientZeroBuffer(ReplayBuffer):
    """Replay storage for EfficientZero / MuZero-style training."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Implement: internal storage for trajectories or transitions, e.g. deque or ring buffer.

    def add(self, transition: Transition) -> None:
        # Implement: store one environment step plus MCTS targets produced by the planner.
        # Consider accepting a richer struct than ``Transition`` if needed.
        raise NotImplementedError("EfficientZeroBuffer.add is not implemented yet.")

    def sample(self, batch_size: int) -> Batch:
        # Implement: return a batch of observations, actions, rewards, policies, and value targets.
        raise NotImplementedError("EfficientZeroBuffer.sample is not implemented yet.")

    def __len__(self) -> int:
        # Implement: return the number of stored entries.
        raise NotImplementedError("EfficientZeroBuffer.__len__ is not implemented yet.")
