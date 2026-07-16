"""Shared agent composition helpers.

Agents declare a composition name registered in ``agents/compositions.py``. Component
construction is delegated to backend registries with explicit dependency injection.
"""

from __future__ import annotations

import gymnasium as gym

from algorl.agents import compositions
from algorl.agents.configs import BaseAgentConfig
from algorl.core.agent_components import AgentComponents
from algorl.core.component_context import ComponentContext
from algorl.core.factory import get_backend, get_component_factory
from algorl.envs.resolve import make_env
from algorl.envs.training_env import TrainingEnv


def compose_agent(
    composition_name: str,
    *,
    env: TrainingEnv | gym.Env,
    config: BaseAgentConfig,
) -> AgentComponents:
    """Build all components declared by a registered agent composition."""
    agent_composition = compositions.registry.create(composition_name)
    resolved_backend = get_backend(config.backend)
    component_factory = get_component_factory(resolved_backend)
    training_env = env if isinstance(env, TrainingEnv) else make_env(env, config=config, seed=config.seed)
    context = ComponentContext(backend=resolved_backend, config=config, env=training_env)

    stub_components: list[str] = []
    for component_type, kind in agent_composition.ordered_slots():
        if component_factory.is_stub(component_type, kind):
            stub_components.append(f"{component_type}={kind!r}")

        component = component_factory.create(component_type, kind, context)
        setattr(context, component_type, component)

    if config.require_implemented and stub_components:
        stub_list = ", ".join(stub_components)
        raise NotImplementedError(
            f"Agent composition {composition_name!r} is not fully implemented. "
            f"Stub component(s): {stub_list}. "
            "Set config.require_implemented=False to construct stub agents for testing."
        )

    if context.world_model is None or context.planner is None or context.learner is None:
        raise RuntimeError(f"Composition {composition_name!r} failed to build required components.")
    if context.replay_buffer is None:
        raise RuntimeError(f"Composition {composition_name!r} failed to build a replay buffer.")

    return AgentComponents(
        backend=context.backend,
        world_model=context.world_model,
        planner=context.planner,
        learner=context.learner,
        replay_buffer=context.replay_buffer,
    )
