"""Search-specific buffers for MCTS-based agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchBuffer:
    """Stores MCTS trees and reanalyze metadata.

    Implement: replace ``Any`` entries with a structured type, for example:
    - root observation or latent state
    - MCTS visit counts / improved policy
    - bootstrap value or value prefix
    - model version used when the entry was created (for reanalyze)
    """

    entries: list[Any] = field(default_factory=list)

    def add(self, entry: Any) -> None:
        # Implement: validate and store one search outcome or reanalyze job.
        self.entries.append(entry)

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)
