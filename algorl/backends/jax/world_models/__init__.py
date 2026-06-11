"""JAX world model registry.

Implement concrete world models in this package, e.g.:
- ``efficient_zero.py`` for EfficientZero / MuZero-style latent dynamics
- ``rssm.py`` for Dreamer / PlaNet
- ``td_mpc.py`` for TD-MPC

Register implementations with ``@registry.register("kind")`` instead of ``if`` chains.
"""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.registry import KindRegistry
from algorl.core.world_model import WorldModel

registry: KindRegistry[WorldModel] = KindRegistry("JAX world model")


class _StubWorldModel(WorldModel):
    """Temporary stand-in until a real world model is registered."""

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


def _register_stub(kind: str) -> None:
    def build(backend: Backend, **kwargs: Any) -> WorldModel:
        return _StubWorldModel(kind, backend)

    registry.register(kind, build)


for _kind in ("efficient_zero", "muzero", "rssm", "td_mpc"):
    _register_stub(_kind)

# When implementing, replace the stub registration with::
#
# registry.replace("efficient_zero", EfficientZeroWorldModel)
#
# or use the decorator on a new module import::
#
# @registry.replace("efficient_zero")
# class EfficientZeroWorldModel(WorldModel):
#     ...
