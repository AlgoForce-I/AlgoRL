"""Episode-level replay storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch, Transition


@dataclass
class EpisodeBuffer:
    """Legacy episode container used before registry-backed buffers."""

    transitions: list[Any] = field(default_factory=list)

    def add(self, transition: Any) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


class EpisodeReplayBuffer(ReplayBuffer):
    """Stores transitions grouped by episode for sequence-based learners."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._episodes: Deque[list[Transition]] = deque()
        self._current_episode: list[Transition] = []
        self._size = 0

    def add(self, transition: Transition) -> None:
        raise NotImplementedError("EpisodeReplayBuffer.add is not implemented yet.")

    def sample(self, batch_size: int) -> Batch:
        raise NotImplementedError("EpisodeReplayBuffer.sample is not implemented yet.")

    def __len__(self) -> int:
        return self._size
