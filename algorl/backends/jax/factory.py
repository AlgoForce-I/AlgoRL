"""JAX component factory."""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.component_factory import ComponentFactory
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.world_model import WorldModel


class JAXComponentFactory(ComponentFactory):
    """Creates JAX implementations of world models, planners, and learners."""

    def create_world_model(self, kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
        from algorl.backends.jax import world_models

        return world_models.registry.create(kind, backend, **kwargs)

    def create_planner(self, kind: str, backend: Backend, **kwargs: Any) -> Planner:
        from algorl.backends.jax import planners

        return planners.registry.create(kind, backend, **kwargs)

    def create_learner(self, kind: str, backend: Backend, **kwargs: Any) -> Learner:
        from algorl.backends.jax import learners

        return learners.registry.create(kind, backend, **kwargs)
