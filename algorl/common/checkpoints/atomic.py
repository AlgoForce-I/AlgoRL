"""Atomic directory replace helpers for checkpoint writes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_parent(path: str | Path) -> Path:
    """Return ``path`` as a Path after creating its parent directory."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def prepare_staging_dir(final_dir: str | Path) -> Path:
    """Create an empty staging directory next to ``final_dir`` (``*.tmp``)."""
    target = Path(final_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def commit_staging_dir(staging_dir: str | Path, final_dir: str | Path) -> Path:
    """Atomically replace ``final_dir`` with ``staging_dir`` via rename."""
    staging = Path(staging_dir)
    target = Path(final_dir)
    if not staging.is_dir():
        raise FileNotFoundError(f"Staging directory missing: {staging}")

    backup = target.with_name(target.name + ".bak")
    if backup.exists():
        shutil.rmtree(backup)

    if target.exists():
        os.replace(target, backup)
    os.replace(staging, target)
    if backup.exists():
        shutil.rmtree(backup)
    return target
