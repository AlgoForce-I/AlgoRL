"""Episode-level storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeBuffer:
    """Stores transitions for a single episode."""

    transitions: list[Any] = field(default_factory=list)

    def add(self, transition: Any) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)
