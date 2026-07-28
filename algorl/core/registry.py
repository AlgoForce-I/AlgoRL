"""Registry for mapping string kinds to concrete builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
Builder = Callable[..., T]


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    """One registered builder and its implementation status."""

    builder: Builder[T]
    stub: bool = False


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
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(self, kind: str, builder: Builder[T], *, stub: bool = False) -> Builder[T]:
        if kind in self._entries:
            raise ValueError(f"{self._component_name} kind {kind!r} is already registered.")
        self._entries[kind] = RegistryEntry(builder=builder, stub=stub)
        return builder

    def is_stub(self, kind: str) -> bool:
        try:
            return self._entries[kind].stub
        except KeyError as error:
            raise ValueError(f"Unknown {self._component_name} kind {kind!r}.") from error

    def create(self, kind: str, *args, **kwargs) -> T:
        try:
            entry = self._entries[kind]
        except KeyError as error:
            available = ", ".join(sorted(self._entries))
            raise ValueError(
                f"Unknown {self._component_name} kind {kind!r}. Available: {available}"
            ) from error
        return entry.builder(*args, **kwargs)

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(self._entries)
