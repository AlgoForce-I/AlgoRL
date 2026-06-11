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

from algorl.core.backend import Backend
from algorl.core.planner import Planner
from algorl.core.registry import KindRegistry
from algorl.core.types import Action, Observation

registry: KindRegistry[Planner] = KindRegistry("JAX planner")


class _StubPlanner(Planner):
    """Temporary stand-in until a real planner is registered."""

    def __init__(self, kind: str, backend: Backend) -> None:
        self.kind = kind
        self.backend = backend

    def search(self, observation: Observation, **kwargs: Any) -> Action:
        # Implement: use the world model (and MCTS / imagination / CEM / MPC) to choose an action.
        # Must return a valid action for ``env.action_space``.
        raise NotImplementedError(f"JAX planner {self.kind!r} is not implemented yet.")


def _register_stub(kind: str) -> None:
    def build(backend: Backend, **kwargs: Any) -> Planner:
        return _StubPlanner(kind, backend)

    registry.register(kind, build)


for _kind in ("mcts", "imagination", "cem", "mpc"):
    _register_stub(_kind)

# When implementing::
#
# @registry.register("mcts")
# class MCTSPlanner(Planner):
#     def __init__(self, backend: Backend, **kwargs: Any) -> None:
#         ...
