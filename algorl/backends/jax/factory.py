"""JAX component factory."""

from __future__ import annotations

from algorl.backends.jax import learners, planners, world_models
from algorl.buffers import buffer_registry
from algorl.core.component_factory import ComponentFactory


class JAXComponentFactory(ComponentFactory):
    """Creates JAX implementations of world models, planners, learners, and buffers."""

    component_registries = {
        "world_model": world_models.registry,
        "planner": planners.registry,
        "learner": learners.registry,
        "replay_buffer": buffer_registry,
    }
