"""EfficientZero replay buffer (stub).

Implement this buffer before wiring algorithm-specific EfficientZero training.
It should store everything the learner needs for one training step.
"""

from __future__ import annotations

from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch, Transition


class EfficientZeroReplayBuffer(ReplayBuffer):
    """Replay storage for EfficientZero / MuZero-style training."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def add(self, transition: Transition) -> None:
        raise NotImplementedError("EfficientZeroReplayBuffer.add is not implemented yet.")

    def sample(self, batch_size: int) -> Batch:
        raise NotImplementedError("EfficientZeroReplayBuffer.sample is not implemented yet.")

    def __len__(self) -> int:
        return 0
