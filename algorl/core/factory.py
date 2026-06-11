"""Backend and component factories.

These functions are the single entry point for constructing backend implementations.
Implement new component kinds in the matching ``backends/<backend>/.../create()`` factory.
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
    if backend_name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend {backend_name!r}. Available: {available}")

    return BACKENDS[backend_name]()


def create_world_model(kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
    """Construct a world model implementation for ``backend``."""
    if backend.name == "jax":
        from algorl.backends.jax import world_models as jax_world_models

        return jax_world_models.create(kind, backend, **kwargs)

    if backend.name == "torch":
        from algorl.backends.torch import world_models as torch_world_models

        return torch_world_models.create(kind, backend, **kwargs)

    raise ValueError(f"No world model factory for backend {backend.name!r}")


def create_planner(kind: str, backend: Backend, **kwargs: Any) -> Planner:
    """Construct a planner implementation for ``backend``."""
    if backend.name == "jax":
        from algorl.backends.jax import planners as jax_planners

        return jax_planners.create(kind, backend, **kwargs)

    if backend.name == "torch":
        from algorl.backends.torch import planners as torch_planners

        return torch_planners.create(kind, backend, **kwargs)

    raise ValueError(f"No planner factory for backend {backend.name!r}")


def create_learner(kind: str, backend: Backend, **kwargs: Any) -> Learner:
    """Construct a learner implementation for ``backend``."""
    if backend.name == "jax":
        from algorl.backends.jax import learners as jax_learners

        return jax_learners.create(kind, backend, **kwargs)

    if backend.name == "torch":
        from algorl.backends.torch import learners as torch_learners

        return torch_learners.create(kind, backend, **kwargs)

    raise ValueError(f"No learner factory for backend {backend.name!r}")
