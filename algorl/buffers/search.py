"""Search-specific replay buffers for MCTS-based agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch, SearchEntry, Transition


@dataclass
class SearchBuffer:
    """Legacy search metadata container used before registry-backed buffers."""

    entries: list[Any] = field(default_factory=list)

    def add(self, entry: Any) -> None:
        self.entries.append(entry)

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)


class SearchReplayBuffer(ReplayBuffer):
    """Stores environment transitions plus MCTS policy and value targets."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._entries: list[SearchEntry] = []
        self._transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        raise NotImplementedError("SearchReplayBuffer.add is not implemented yet.")

    def sample(self, batch_size: int) -> Batch:
        raise NotImplementedError("SearchReplayBuffer.sample is not implemented yet.")

    def __len__(self) -> int:
        return len(self._transitions)
