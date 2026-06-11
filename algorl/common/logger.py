"""Simple training logger."""

from __future__ import annotations

from typing import Any


class Logger:
    """Minimal key-value logger for training metrics."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def record(self, step: int, metrics: dict[str, Any]) -> None:
        self.history.append({"step": step, **metrics})

    def dump(self) -> list[dict[str, Any]]:
        return list(self.history)
