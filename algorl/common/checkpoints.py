"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_checkpoint_dir(path: str | Path) -> Path:
    """Create the parent directory for a checkpoint path if needed."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint payload to disk."""
    import pickle

    checkpoint_path = ensure_checkpoint_dir(path)
    with checkpoint_path.open("wb") as checkpoint_file:
        pickle.dump(payload, checkpoint_file)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint payload from disk.

    Implement: replace pickle with a safer format if needed and validate payload keys
    before restoring model parameters, optimizer state, and training step counters.
    """
    import pickle

    with Path(path).open("rb") as checkpoint_file:
        return pickle.load(checkpoint_file)
