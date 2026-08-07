"""Backend registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from algorl.core.backend import Backend
    from algorl.core.component_factory import ComponentFactory

DEFAULT_BACKEND = "jax"

BACKENDS: dict[str, Type[Backend]] = {}
COMPONENT_FACTORIES: dict[str, Type[ComponentFactory]] = {}


def register_backend(
    name: str,
    backend_cls: Type[Backend],
    component_factory_cls: Type[ComponentFactory],
) -> None:
    """Register a backend and its component factory."""
    BACKENDS[name] = backend_cls
    COMPONENT_FACTORIES[name] = component_factory_cls


def _register_backends() -> None:
    from algorl.backends.jax.backend import JAXBackend
    from algorl.backends.jax.factory import JAXComponentFactory
    from algorl.backends.torch.backend import TorchBackend
    from algorl.backends.torch.factory import TorchComponentFactory

    register_backend("jax", JAXBackend, JAXComponentFactory)
    register_backend("torch", TorchBackend, TorchComponentFactory)


_register_backends()
