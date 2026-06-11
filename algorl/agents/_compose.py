"""Shared agent composition helpers.

Agents declare a composition name registered in ``agents/compositions.py``. Component
construction is delegated to backend registries without conditional routing.
"""

from __future__ import annotations

from typing import Any

from algorl.agents import compositions
from algorl.core.agent_components import AgentComponents
from algorl.core.factory import get_backend, get_component_factory


def compose_agent(composition_name: str, *, backend: str, **kwargs: Any) -> AgentComponents:
    """Build all components declared by a registered agent composition."""
    agent_composition = compositions.registry.create(composition_name)
    resolved_backend = get_backend(backend)
    component_factory = get_component_factory(resolved_backend)

    built_components: dict[str, object] = {}
    for component_type, kind in agent_composition.components:
        built_components[component_type] = component_factory.create(
            component_type,
            kind,
            resolved_backend,
            **kwargs,
        )

    return AgentComponents(
        backend=resolved_backend,
        world_model=built_components[compositions.COMPONENT_WORLD_MODEL],
        planner=built_components[compositions.COMPONENT_PLANNER],
        learner=built_components[compositions.COMPONENT_LEARNER],
    )
