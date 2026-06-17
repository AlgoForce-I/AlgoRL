"""Training callbacks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Callback(Protocol):
    """Hook invoked during training."""

    def on_step(self, step: int, info: dict[str, Any]) -> None: ...


class CallbackList:
    """Run multiple callbacks in sequence."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = callbacks or []

    def on_step(self, step: int, info: dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_step(step, info)
