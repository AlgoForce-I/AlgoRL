"""JAX world model factory.

Implement concrete world models in this package, e.g.:
- ``efficient_zero.py`` for EfficientZero / MuZero-style latent dynamics
- ``rssm.py`` for Dreamer / PlaNet
- ``td_mpc.py`` for TD-MPC

Register each kind in ``create()`` below.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.world_model import WorldModel

# Maps factory ``kind`` strings to the module that should implement them.
# Implement: backends/jax/world_models/<module>.py and import it from ``create()``.
WORLD_MODEL_KINDS = {
    "efficient_zero": "MuZero-style representation, dynamics, reward, and value heads.",
    "muzero": "Same latent structure as EfficientZero; can share code initially.",
    "rssm": "Recurrent state-space model for Dreamer and PlaNet.",
    "td_mpc": "Latent model used by TD-MPC / TD-MPC2.",
}


class _StubWorldModel:
    """Temporary stand-in until a real world model is registered in ``create()``."""

    def __init__(self, kind: str, backend: Backend) -> None:
        self.kind = kind
        self.backend = backend

    def encode(self, observation: Any) -> Any:
        # Implement: map a Gymnasium observation to a latent state (representation network).
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def transition(self, latent_state: Any, action: Any) -> Any:
        # Implement: predict the next latent state given the current latent state and action.
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def reward(self, latent_state: Any, action: Any) -> Any:
        # Implement: predict immediate reward from latent state and action (used during MCTS).
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def value(self, latent_state: Any) -> Any:
        # Implement: predict state value from latent state (used by MCTS and training targets).
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")


def create(kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
    """Return a backend world model for ``kind``.

    Implement: replace the stub branch with imports from concrete modules, e.g.::

        if kind == "efficient_zero":
            from algorl.backends.jax.world_models.efficient_zero import EfficientZeroWorldModel
            return EfficientZeroWorldModel(backend, **kwargs)
    """
    if kind not in WORLD_MODEL_KINDS:
        available = ", ".join(sorted(WORLD_MODEL_KINDS))
        raise ValueError(f"Unknown world model kind {kind!r}. Available: {available}")

    return _StubWorldModel(kind, backend)
