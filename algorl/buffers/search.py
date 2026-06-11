"""Search-specific buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchBuffer:
    """Stores MCTS trees and reanalyze metadata."""

    entries: list[Any] = field(default_factory=list)

    def add(self, entry: Any) -> None:
        self.entries.append(entry)

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)
