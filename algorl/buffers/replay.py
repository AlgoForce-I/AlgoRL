"""Simple NumPy replay buffer.

This generic buffer is enough for early experiments. Search-based agents such as
MuZero and EfficientZero will likely need a dedicated buffer that also stores:
- MCTS policy targets
- n-step or model-based value targets
- optional search metadata for reanalyze

Implement algorithm-specific buffers in this package, e.g. ``efficient_zero.py``.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from algorl.core.replay_buffer import ReplayBuffer as BaseReplayBuffer
from algorl.core.types import Batch, Transition


class ReplayBuffer(BaseReplayBuffer):
    """Fixed-size FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._storage.append(transition)

    def sample(self, batch_size: int) -> Batch:
        if batch_size > len(self._storage):
            raise ValueError(
                f"Requested batch size {batch_size} exceeds buffer size {len(self._storage)}."
            )

        import random

        transitions = random.sample(list(self._storage), batch_size)
        return Batch(data={"transitions": transitions})

    def __len__(self) -> int:
        return len(self._storage)
