"""Backend component factory abstract base class."""

from __future__ import annotations

from typing import Any, ClassVar

from algorl.core.component_context import ComponentContext
from algorl.core.registry import KindRegistry


class ComponentFactory:
    """Creates algorithm components for one backend implementation."""

    component_registries: ClassVar[dict[str, KindRegistry[Any]]] = {}

    def create(self, component_type: str, kind: str, context: ComponentContext) -> Any:
        try:
            component_registry = self.component_registries[component_type]
        except KeyError as error:
            available = ", ".join(sorted(self.component_registries))
            raise ValueError(
                f"Unknown component type {component_type!r}. Available: {available}"
            ) from error

        return component_registry.create(kind, context)

    def is_stub(self, component_type: str, kind: str) -> bool:
        try:
            component_registry = self.component_registries[component_type]
        except KeyError as error:
            available = ", ".join(sorted(self.component_registries))
            raise ValueError(
                f"Unknown component type {component_type!r}. Available: {available}"
            ) from error
        return component_registry.is_stub(kind)
