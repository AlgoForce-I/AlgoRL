"""Registry for mapping string kinds to concrete builders."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")
Builder = Callable[..., T]


class KindRegistry(Generic[T]):
    """Maps a string kind to a builder callable.

    Use this instead of long ``if kind == ...`` chains. Register implementations with
    ``register()`` or the decorator form::

        @registry.register("efficient_zero")
        class EfficientZeroWorldModel(WorldModel):
            ...
    """

    def __init__(self, component_name: str) -> None:
        self._component_name = component_name
        self._builders: dict[str, Builder[T]] = {}

    def register(self, kind: str, builder: Builder[T]) -> Builder[T]:
        if kind in self._builders:
            raise ValueError(f"{self._component_name} kind {kind!r} is already registered.")
        self._builders[kind] = builder
        return builder

    def replace(self, kind: str, builder: Builder[T]) -> Builder[T]:
        """Replace an existing registration, e.g. swap a stub for a real implementation."""
        self._builders[kind] = builder
        return builder

    def create(self, kind: str, *args, **kwargs) -> T:
        try:
            builder = self._builders[kind]
        except KeyError as error:
            available = ", ".join(sorted(self._builders))
            raise ValueError(
                f"Unknown {self._component_name} kind {kind!r}. Available: {available}"
            ) from error
        return builder(*args, **kwargs)

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(self._builders)
