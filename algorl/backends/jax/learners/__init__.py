"""JAX learner factory.

Implement concrete learners in this package, e.g.:
- ``efficient_zero.py`` for EfficientZero losses and parameter updates
- ``muzero.py`` for MuZero losses
- ``alphazero.py`` for AlphaZero policy/value losses
- ``dreamer.py`` for DreamerV3 losses
- ``planet.py`` for PlaNet losses
- ``td_mpc.py`` for TD-MPC losses

Register each kind in ``create()`` below.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.learner import Learner
from algorl.core.replay_buffer import ReplayBuffer

# Maps factory ``kind`` strings to the training behaviour that must be implemented.
LEARNER_KINDS = {
    "efficient_zero": "Policy, value, reward, and self-supervised losses; optional reanalyze.",
    "muzero": "Policy, value, and reward losses from MCTS targets.",
    "alphazero": "Policy and value losses from MCTS targets (no learned dynamics).",
    "dreamer": "World model, actor, and critic losses from imagined trajectories.",
    "planet": "ELBO-style world model loss and CEM-related training.",
    "td_mpc": "TD-MPC latent consistency, reward, and value losses.",
}


class _StubLearner:
    """Temporary stand-in until a real learner is registered in ``create()``."""

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


def create(kind: str, backend: Backend, **kwargs: Any) -> Learner:
    """Return a backend learner for ``kind``.

    Implement: replace the stub branch with imports from concrete modules, e.g.::

        if kind == "efficient_zero":
            from algorl.backends.jax.learners.efficient_zero import EfficientZeroLearner
            return EfficientZeroLearner(backend, **kwargs)
    """
    if kind not in LEARNER_KINDS:
        available = ", ".join(sorted(LEARNER_KINDS))
        raise ValueError(f"Unknown learner kind {kind!r}. Available: {available}")

    return _StubLearner(kind, backend)
