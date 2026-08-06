"""Checkpoint directory manifest + JSON artifact helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "agent",
    "step",
    "task_id",
    "created_at",
    "artifacts",
)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)!r}.")
    return payload


def build_manifest(
    *,
    agent: str,
    step: int,
    task_id: int | None = None,
    artifacts: dict[str, str] | None = None,
    config_hash: str | None = None,
    tag: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema-versioned checkpoint manifest."""
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "agent": agent,
        "step": int(step),
        "task_id": None if task_id is None else int(task_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": dict(artifacts or {}),
    }
    if config_hash is not None:
        manifest["config_hash"] = config_hash
    if tag is not None:
        manifest["tag"] = tag
    if extra:
        manifest.update(extra)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate required keys and schema version; return the same dict."""
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"Checkpoint manifest missing keys: {missing}.")
    version = int(manifest["schema_version"])
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema_version={version}; "
            f"this build expects {SCHEMA_VERSION}."
        )
    if not isinstance(manifest["artifacts"], dict):
        raise TypeError("manifest['artifacts'] must be a dict of relative paths.")
    return manifest


def write_manifest(directory: str | Path, manifest: dict[str, Any]) -> Path:
    """Validate and write ``manifest.json`` under ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validate_manifest(manifest)
    path = directory / "manifest.json"
    write_json(path, manifest)
    return path


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """Load and validate ``manifest.json`` from a checkpoint directory."""
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint manifest: {path}")
    return validate_manifest(read_json(path))
