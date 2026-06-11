"""JAX planner factory.

Implement concrete planners in this package, e.g.:
- ``mcts.py`` for MuZero / EfficientZero / AlphaZero search (use MCTX)
- ``imagination.py`` for Dreamer latent rollouts
- ``cem.py`` for PlaNet cross-entropy method planning
- ``mpc.py`` for TD-MPC shooting-based control

Register each kind in ``create()`` below.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.planner import Planner
from algorl.core.types import Action, Observation

# Maps factory ``kind`` strings to the behaviour that must be implemented.
PLANNER_KINDS = {
    "mcts": "Run MCTS in latent space and return the action with highest visit count.",
    "imagination": "Roll out the world model in latent space and optimize actions (Dreamer).",
    "cem": "Sample action sequences, evaluate with the model, refit with CEM (PlaNet).",
    "mpc": "Plan actions with model predictive control (TD-MPC).",
}


class _StubPlanner:
    """Temporary stand-in until a real planner is registered in ``create()``."""

    def __init__(self, kind: str, backend: Backend) -> None:
        self.kind = kind
        self.backend = backend

    def search(self, observation: Observation, **kwargs: Any) -> Action:
        # Implement: use the world model (and MCTS / imagination / CEM / MPC) to choose an action.
        # Must return a valid action for ``env.action_space``.
        raise NotImplementedError(f"JAX planner {self.kind!r} is not implemented yet.")


def create(kind: str, backend: Backend, **kwargs: Any) -> Planner:
    """Return a backend planner for ``kind``.

    Implement: replace the stub branch with imports from concrete modules, e.g.::

        if kind == "mcts":
            from algorl.backends.jax.planners.mcts import MCTSPlanner
            return MCTSPlanner(backend, **kwargs)
    """
    if kind not in PLANNER_KINDS:
        available = ", ".join(sorted(PLANNER_KINDS))
        raise ValueError(f"Unknown planner kind {kind!r}. Available: {available}")

    return _StubPlanner(kind, backend)
