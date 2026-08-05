"""JAX world model registry.

Implement concrete world models in this package, e.g.:
- ``efficientzero/`` for EfficientZero / MuZero-style latent dynamics
- ``rssm.py`` for Dreamer / PlaNet
- ``td_mpc.py`` for TD-MPC

Register implementations with ``@registry.register("kind")`` instead of ``if`` chains.
"""

from __future__ import annotations

from typing import Any

from algorl.core.component_context import ComponentContext
from algorl.core.registry import KindRegistry
from algorl.core.types import Action, LatentState, Observation
from algorl.core.world_model import WorldModel

registry: KindRegistry[WorldModel] = KindRegistry("JAX world model")


class _StubWorldModel(WorldModel):
    """Temporary stand-in until a real world model is registered."""

    def __init__(self, kind: str, context: ComponentContext) -> None:
        self.kind = kind
        self.backend = context.backend
        self.config = context.config

    def encode(self, observation: Observation) -> LatentState:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def transition(self, latent_state: LatentState, action: Action) -> LatentState:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def reward(self, latent_state: LatentState, action: Action) -> Any:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def value(self, latent_state: LatentState) -> Any:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")


class _NullWorldModel(WorldModel):
    """Placeholder for agents such as AlphaZero that do not use learned dynamics."""

    def __init__(self, context: ComponentContext) -> None:
        self.backend = context.backend
        self.config = context.config

    def encode(self, observation: Observation) -> LatentState:
        raise NotImplementedError("AlphaZero does not use a learned world model.")

    def transition(self, latent_state: LatentState, action: Action) -> LatentState:
        raise NotImplementedError("AlphaZero does not use a learned world model.")

    def reward(self, latent_state: LatentState, action: Action) -> Any:
        raise NotImplementedError("AlphaZero does not use a learned world model.")

    def value(self, latent_state: LatentState) -> Any:
        raise NotImplementedError("AlphaZero does not use a learned world model.")


def _register_stub(kind: str) -> None:
    def build(context: ComponentContext) -> WorldModel:
        return _StubWorldModel(kind, context)

    registry.register(kind, build, stub=True)


registry.register("none", _NullWorldModel, stub=False)

from algorl.backends.jax.world_models.efficientzero import build_efficient_zero_world_model
from algorl.backends.jax.world_models.hypercez import build_hyper_cez_world_model

registry.register("efficient_zero", build_efficient_zero_world_model, stub=False)
registry.register("hyper_cez", build_hyper_cez_world_model, stub=False)

for _kind in ("muzero", "rssm", "td_mpc"):
    _register_stub(_kind)
