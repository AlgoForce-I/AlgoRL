"""Protocol for components that participate in multi-file run checkpoints."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Checkpointable(Protocol):
    """Objects that can export / restore a structured checkpoint payload."""

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a JSON/pytree-serializable state dict."""

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore from a payload produced by :meth:`checkpoint_state`."""
