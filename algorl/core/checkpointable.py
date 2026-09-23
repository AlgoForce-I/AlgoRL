"""Protocol for components that participate in multi-file run checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Checkpointable(Protocol):
    """A component that persists itself as one directory inside a run checkpoint.

    :func:`~algorl.common.checkpoints.run.save_run_checkpoint` gives each
    checkpointable component its own subdirectory and records it in the manifest,
    so components own their format and can evolve it independently of the run
    layout. The learner is required to satisfy this protocol; the replay buffer
    is persisted when it does.

    ``isinstance`` checks against this protocol verify that both methods exist,
    not their signatures, so implementations must accept a single directory.
    """

    def save(self, directory: str | Path) -> None:
        """Write this component's state into ``directory`` (created if absent)."""

    def load(self, directory: str | Path) -> None:
        """Restore the state previously written by :meth:`save`."""
