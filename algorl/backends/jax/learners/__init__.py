"""JAX learner registry.

Implement concrete learners in this package, e.g.:
- ``efficient_zero.py`` for EfficientZero losses and parameter updates
- ``muzero.py`` for MuZero losses
- ``alphazero.py`` for AlphaZero policy/value losses
- ``dreamer.py`` for DreamerV3 losses
- ``planet.py`` for PlaNet losses
- ``td_mpc.py`` for TD-MPC losses

Register implementations with ``@registry.register("kind")`` instead of ``if`` chains.
"""

from __future__ import annotations

from algorl.backends.jax.learners.efficient_zero import (
    EfficientZeroLearner,
    build_efficient_zero_learner,
)
from algorl.core.component_context import ComponentContext
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.registry import KindRegistry
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.world_model import WorldModel

registry: KindRegistry[Learner] = KindRegistry("JAX learner")


class _StubLearner(Learner):
    """Temporary stand-in until a real learner is registered."""

    def __init__(self, kind: str, context: ComponentContext) -> None:
        self.kind = kind
        self.backend = context.backend
        self.config = context.config
        self.world_model: WorldModel | None = context.world_model
        self.planner: Planner | None = context.planner

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        raise NotImplementedError(f"JAX learner {self.kind!r} is not implemented yet.")


def _register_stub(kind: str) -> None:
    def build(context: ComponentContext) -> Learner:
        if kind != "alphazero" and context.world_model is None:
            raise RuntimeError(f"Learner {kind!r} requires a world model in the build context.")
        if context.planner is None:
            raise RuntimeError(f"Learner {kind!r} requires a planner in the build context.")
        return _StubLearner(kind, context)

    registry.register(kind, build, stub=True)


registry.register("efficient_zero", build_efficient_zero_learner, stub=False)

for _kind in ("muzero", "alphazero", "dreamer", "planet", "td_mpc"):
    _register_stub(_kind)
