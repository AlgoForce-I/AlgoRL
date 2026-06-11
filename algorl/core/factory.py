"""Backend and component factories.

These functions are the single public entry point for constructing backend implementations.
Backends register a ``ComponentFactory`` in ``algorl.backends.registry``; no ``if backend``
branches should be added here.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.world_model import WorldModel


def get_backend(name: str = "jax") -> Backend:
    """Return a backend instance by name."""
    from algorl.backends.registry import BACKENDS, DEFAULT_BACKEND

    backend_name = name or DEFAULT_BACKEND
    try:
        backend_cls = BACKENDS[backend_name]
    except KeyError as error:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend {backend_name!r}. Available: {available}") from error

    return backend_cls()


def _get_component_factory(backend: Backend):
    from algorl.backends.registry import COMPONENT_FACTORIES

    try:
        factory_cls = COMPONENT_FACTORIES[backend.name]
    except KeyError as error:
        raise ValueError(f"No component factory registered for backend {backend.name!r}.") from error

    return factory_cls()


def create_world_model(kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
    """Construct a world model implementation for ``backend``."""
    return _get_component_factory(backend).create_world_model(kind, backend, **kwargs)


def create_planner(kind: str, backend: Backend, **kwargs: Any) -> Planner:
    """Construct a planner implementation for ``backend``."""
    return _get_component_factory(backend).create_planner(kind, backend, **kwargs)


def create_learner(kind: str, backend: Backend, **kwargs: Any) -> Learner:
    """Construct a learner implementation for ``backend``."""
    return _get_component_factory(backend).create_learner(kind, backend, **kwargs)
