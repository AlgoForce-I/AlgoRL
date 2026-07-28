"""Backend and component factories.

Single public entry point for constructing backend implementations. Routing is handled
entirely through registries registered in ``algorl.backends.registry``.
"""

from __future__ import annotations

from algorl.core.backend import Backend


def get_backend(name: str = "jax") -> Backend:
    """Return a backend instance by name."""
    from algorl.backends.registry import BACKENDS, DEFAULT_BACKEND

    backend_name = name or DEFAULT_BACKEND
    try:
        backend_cls = BACKENDS[backend_name]
    except KeyError as error:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend {backend_name!r}. Available: {available}") from error

    return backend_cls()


def get_component_factory(backend: Backend):
    from algorl.backends.registry import COMPONENT_FACTORIES

    try:
        factory_cls = COMPONENT_FACTORIES[backend.name]
    except KeyError as error:
        raise ValueError(f"No component factory registered for backend {backend.name!r}.") from error

    return factory_cls()
