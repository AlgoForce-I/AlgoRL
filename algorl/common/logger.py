"""Simple training logger."""

from __future__ import annotations

from typing import Any

DEFAULT_HISTORY_LIMIT = 20_000


class Logger:
    """Minimal key-value logger for training metrics.

    ``history`` keeps only the most recent ``history_limit`` records. Long runs
    record several entries per environment step and the history is also
    serialized into every checkpoint, so an unbounded list grows without limit.
    Pass ``history_limit=None`` to keep everything.
    """

    def __init__(self, *, history_limit: int | None = DEFAULT_HISTORY_LIMIT) -> None:
        self.history: list[dict[str, Any]] = []
        self._history_limit = (
            None if history_limit is None else max(1, int(history_limit))
        )

    def record(self, step: int, metrics: dict[str, Any]) -> None:
        self.history.append({"step": step, **metrics})
        self._trim_history()

    def dump(self) -> list[dict[str, Any]]:
        return list(self.history)

    def _trim_history(self) -> None:
        limit = self._history_limit
        # Trim in blocks: dropping one entry per record would memmove the whole
        # list on every logged step.
        if limit is None or len(self.history) <= 2 * limit:
            return
        del self.history[: len(self.history) - limit]
