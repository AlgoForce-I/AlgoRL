"""JAX learner factory."""

from __future__ import annotations

from typing import Any

from algorl.core.backend import Backend
from algorl.core.learner import Learner
from algorl.core.replay_buffer import ReplayBuffer


class _StubLearner:
    """Placeholder learner until training loops land."""

    def __init__(self, kind: str, backend: Backend) -> None:
        self.kind = kind
        self.backend = backend

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        raise NotImplementedError(f"JAX learner {self.kind!r} is not implemented yet.")


def create(kind: str, backend: Backend, **kwargs: Any) -> Learner:
    return _StubLearner(kind, backend)
