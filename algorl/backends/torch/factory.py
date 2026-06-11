"""PyTorch component factory (stub)."""

from __future__ import annotations

from algorl.backends.torch import learners, planners, world_models
from algorl.core.component_factory import ComponentFactory


class TorchComponentFactory(ComponentFactory):
    """Creates PyTorch implementations once the torch backend is added."""

    component_registries = {
        "world_model": world_models.registry,
        "planner": planners.registry,
        "learner": learners.registry,
    }
