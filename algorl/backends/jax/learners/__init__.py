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

from typing import Any

from algorl.core.backend import Backend
from algorl.core.learner import Learner
from algorl.core.registry import KindRegistry
from algorl.core.replay_buffer import ReplayBuffer

registry: KindRegistry[Learner] = KindRegistry("JAX learner")


class _StubLearner(Learner):
    """Temporary stand-in until a real learner is registered."""

    def __init__(self, kind: str, backend: Backend) -> None:
        self.kind = kind
        self.backend = backend

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        # Implement:
        # 1. Sample a batch from the replay buffer.
        # 2. Compute losses against stored targets (MCTS policy, n-step value, etc.).
        # 3. Apply an Optax optimizer step to the world model / policy parameters.
        # 4. Return scalar metrics such as total loss and individual loss terms.
        raise NotImplementedError(f"JAX learner {self.kind!r} is not implemented yet.")


def _register_stub(kind: str) -> None:
    def build(backend: Backend, **kwargs: Any) -> Learner:
        return _StubLearner(kind, backend)

    registry.register(kind, build)


for _kind in ("efficient_zero", "muzero", "alphazero", "dreamer", "planet", "td_mpc"):
    _register_stub(_kind)

# When implementing::
#
# @registry.register("efficient_zero")
# class EfficientZeroLearner(Learner):
#     def __init__(self, backend: Backend, **kwargs: Any) -> None:
#         ...
