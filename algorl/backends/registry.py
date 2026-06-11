"""Backend registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from algorl.core.backend import Backend

DEFAULT_BACKEND = "jax"

BACKENDS: dict[str, Callable[[], Backend]] = {}


def _register_backends() -> None:
    from algorl.backends.jax.backend import JAXBackend
    from algorl.backends.torch.backend import TorchBackend

    BACKENDS["jax"] = JAXBackend
    BACKENDS["torch"] = TorchBackend


_register_backends()
