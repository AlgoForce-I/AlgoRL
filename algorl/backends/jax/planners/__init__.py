"""JAX planner registry.

Implement concrete planners in this package, e.g.:
- ``mcts.py`` for MuZero / EfficientZero / AlphaZero search (use MCTX)
- ``imagination.py`` for Dreamer latent rollouts
- ``cem.py`` for PlaNet cross-entropy method planning
- ``mpc.py`` for TD-MPC shooting-based control

Register implementations with ``@registry.register("kind")`` instead of ``if`` chains.
"""

from __future__ import annotations

from typing import Any

from algorl.core.component_context import ComponentContext
from algorl.core.planner import Planner
from algorl.core.registry import KindRegistry
from algorl.core.types import Action, Observation
from algorl.core.world_model import WorldModel

registry: KindRegistry[Planner] = KindRegistry("JAX planner")


class _StubPlanner(Planner):
    """Temporary stand-in until a real planner is registered."""

    def __init__(self, kind: str, context: ComponentContext) -> None:
        self.kind = kind
        self.backend = context.backend
        self.config = context.config
        self.world_model: WorldModel | None = context.world_model
        self.env = context.env

    def search(self, observation: Observation, **kwargs: Any) -> Action:
        raise NotImplementedError(f"JAX planner {self.kind!r} is not implemented yet.")


def _register_stub(kind: str) -> None:
    def build(context: ComponentContext) -> Planner:
        if context.world_model is None and kind not in {"mcts", "alphazero"}:
            raise RuntimeError(f"Planner {kind!r} requires a world model in the build context.")
        return _StubPlanner(kind, context)

    registry.register(kind, build, stub=True)


def _build_alphazero_planner(context: ComponentContext) -> Planner:
    from algorl.backends.jax.planners.mcts.alphazero import build_alphazero_planner

    return build_alphazero_planner(context)


def _build_efficient_zero_planner(context: ComponentContext) -> Planner:
    from algorl.backends.jax.planners.mcts.efficientzero import build_efficient_zero_planner

    return build_efficient_zero_planner(context)


registry.register("alphazero", _build_alphazero_planner, stub=False)
registry.register("mcts", _build_efficient_zero_planner, stub=False)

for _kind in ("imagination", "cem", "mpc"):
    _register_stub(_kind)
