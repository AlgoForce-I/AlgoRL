"""Multi-file run checkpoints (directory + manifest + pytree/JSON artifacts).

This package replaces whole-object pickling. A checkpoint is a directory::

    my_ckpt/
      manifest.json
      config.json          # optional
      loop.json            # optional
      learner/             # pytree store
      buffer/              # buffer-specific layout (later)
      ...

Use :func:`open_checkpoint_write` / :func:`commit_checkpoint` for atomic writes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from algorl.common.checkpoints.atomic import (
    commit_staging_dir,
    ensure_parent,
    prepare_staging_dir,
)
from algorl.common.checkpoints.manifest import (
    SCHEMA_VERSION,
    build_manifest,
    read_json,
    read_manifest,
    validate_manifest,
    write_json,
    write_manifest,
)
from algorl.common.checkpoints.config_diff import (
    RESUME_ALLOWLIST,
    RESUME_STRUCTURAL,
    split_resume_overrides,
)
from algorl.common.checkpoints.pytree_store import (
    load_pytree,
    load_pytree_as_jax,
    save_pytree,
)
from algorl.common.checkpoints.run import (
    load_run_checkpoint,
    prune_step_checkpoints,
    save_run_checkpoint,
    write_resume_overrides,
)

__all__ = [
    "RESUME_ALLOWLIST",
    "RESUME_STRUCTURAL",
    "SCHEMA_VERSION",
    "build_manifest",
    "commit_checkpoint",
    "commit_staging_dir",
    "ensure_checkpoint_dir",
    "ensure_parent",
    "load_checkpoint",
    "load_pytree",
    "load_pytree_as_jax",
    "load_run_checkpoint",
    "open_checkpoint_write",
    "prepare_staging_dir",
    "prune_step_checkpoints",
    "read_json",
    "read_manifest",
    "save_checkpoint",
    "save_pytree",
    "save_run_checkpoint",
    "split_resume_overrides",
    "validate_manifest",
    "write_json",
    "write_manifest",
    "write_resume_overrides",
]


def ensure_checkpoint_dir(path: str | Path) -> Path:
    """Create parent dirs for a checkpoint path (file or directory)."""
    return ensure_parent(path)


def open_checkpoint_write(final_dir: str | Path) -> Path:
    """Create a staging directory for an atomic checkpoint write."""
    return prepare_staging_dir(final_dir)


def commit_checkpoint(staging_dir: str | Path, final_dir: str | Path) -> Path:
    """Publish a staged checkpoint directory to ``final_dir``."""
    return commit_staging_dir(staging_dir, final_dir)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Deprecated pickle stub used by the legacy TrainingLoop hook.

    Prefer directory checkpoints via :func:`open_checkpoint_write` and
    :func:`save_pytree`. This will be removed once the loop is migrated.
    """
    import pickle

    warnings.warn(
        "algorl.common.checkpoints.save_checkpoint(pickle) is deprecated; "
        "use multi-file directory checkpoints.",
        DeprecationWarning,
        stacklevel=2,
    )
    checkpoint_path = ensure_checkpoint_dir(path)
    with checkpoint_path.open("wb") as checkpoint_file:
        pickle.dump(payload, checkpoint_file)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Deprecated pickle loader complementary to :func:`save_checkpoint`."""
    import pickle

    warnings.warn(
        "algorl.common.checkpoints.load_checkpoint(pickle) is deprecated; "
        "use read_manifest / load_pytree on a checkpoint directory.",
        DeprecationWarning,
        stacklevel=2,
    )
    with Path(path).open("rb") as checkpoint_file:
        return pickle.load(checkpoint_file)
