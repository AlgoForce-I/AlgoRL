"""JAX world model factory."""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.world_model import WorldModel


class _StubWorldModel:
    """Placeholder world model until algorithm implementations land."""

    def __init__(self, kind: str, backend: Backend) -> None:
        self.kind = kind
        self.backend = backend

    def encode(self, observation: Any) -> Any:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def transition(self, latent_state: Any, action: Any) -> Any:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def reward(self, latent_state: Any, action: Any) -> Any:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")

    def value(self, latent_state: Any) -> Any:
        raise NotImplementedError(f"JAX world model {self.kind!r} is not implemented yet.")


def create(kind: str, backend: Backend, **kwargs: Any) -> WorldModel:
    return _StubWorldModel(kind, backend)
