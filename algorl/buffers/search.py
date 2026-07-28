"""Search-specific replay buffers for MCTS-based agents."""

from __future__ import annotations

from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Batch, SearchEntry, Transition


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
