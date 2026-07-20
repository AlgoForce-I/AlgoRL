"""Backend-agnostic core abstractions.

Exports are lazy so importing leaf modules (e.g. ``algorl.core.types``) does not
pull ``BaseAgent`` / env wiring and create circular imports.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Backend": ("algorl.core.backend", "Backend"),
    "BaseAgent": ("algorl.core.agent", "BaseAgent"),
    "BatchedPlanner": ("algorl.core.planner", "BatchedPlanner"),
    "Learner": ("algorl.core.learner", "Learner"),
    "Planner": ("algorl.core.planner", "Planner"),
    "ReplayBuffer": ("algorl.core.replay_buffer", "ReplayBuffer"),
    "WorldModel": ("algorl.core.world_model", "WorldModel"),
    "get_backend": ("algorl.core.factory", "get_backend"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    return getattr(importlib.import_module(module_name), attr_name)


__all__ = [
    "Backend",
    "BaseAgent",
    "BatchedPlanner",
    "Learner",
    "Planner",
    "ReplayBuffer",
    "WorldModel",
    "get_backend",
]
