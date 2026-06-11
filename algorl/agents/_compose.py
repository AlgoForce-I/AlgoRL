"""Shared agent composition helpers.

Agents call ``compose_agent()`` to build backend-specific components via
``core/factory.py``. Implementations are registered in each backend's
``KindRegistry`` rather than selected with ``if`` chains.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.factory import create_learner, create_planner, create_world_model, get_backend
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.world_model import WorldModel


def compose_agent(
    *,
    backend: str,
    world_model_kind: str | None = None,
    planner_kind: str | None = None,
    learner_kind: str,
    **kwargs: Any,
) -> tuple[Backend, WorldModel | None, Planner | None, Learner]:
    """Wire backend-specific components for an agent."""
    resolved_backend = get_backend(backend)
    world_model = (
        create_world_model(world_model_kind, resolved_backend, **kwargs)
        if world_model_kind is not None
        else None
    )
    planner = (
        create_planner(planner_kind, resolved_backend, **kwargs)
        if planner_kind is not None
        else None
    )
    learner = create_learner(learner_kind, resolved_backend, **kwargs)
    return resolved_backend, world_model, planner, learner
