"""Registered agent composition specs."""

from __future__ import annotations

from dataclasses import dataclass

from algorl.core.registry import KindRegistry

COMPONENT_WORLD_MODEL = "world_model"
COMPONENT_PLANNER = "planner"
COMPONENT_LEARNER = "learner"


@dataclass(frozen=True)
class AgentComposition:
    """Ordered component slots resolved through backend registries."""

    components: tuple[tuple[str, str], ...]


registry: KindRegistry[AgentComposition] = KindRegistry("agent composition")


def _register(
    name: str,
    *,
    world_model_kind: str,
    planner_kind: str,
    learner_kind: str,
) -> None:
    registry.register(
        name,
        lambda world_model_kind=world_model_kind, planner_kind=planner_kind, learner_kind=learner_kind: AgentComposition(
            components=(
                (COMPONENT_WORLD_MODEL, world_model_kind),
                (COMPONENT_PLANNER, planner_kind),
                (COMPONENT_LEARNER, learner_kind),
            )
        ),
    )


_register(
    "efficient_zero",
    world_model_kind="efficient_zero",
    planner_kind="mcts",
    learner_kind="efficient_zero",
)
_register(
    "muzero",
    world_model_kind="muzero",
    planner_kind="mcts",
    learner_kind="muzero",
)
_register(
    "alphazero",
    world_model_kind="none",
    planner_kind="mcts",
    learner_kind="alphazero",
)
_register(
    "dreamer_v3",
    world_model_kind="rssm",
    planner_kind="imagination",
    learner_kind="dreamer",
)
_register(
    "planet",
    world_model_kind="rssm",
    planner_kind="cem",
    learner_kind="planet",
)
_register(
    "td_mpc",
    world_model_kind="td_mpc",
    planner_kind="mpc",
    learner_kind="td_mpc",
)
