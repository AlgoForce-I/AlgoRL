"""Configuration helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass config to a plain dictionary."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    return asdict(config)
