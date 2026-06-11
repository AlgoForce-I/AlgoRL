"""Backend abstract base class for framework isolation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Backend(ABC):
    """Minimal interface that every ML backend must implement explicitly."""

    name: str

    @abstractmethod
    def array(self, value: Any) -> Any:
        """Convert a Python or NumPy value to a backend array."""

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create a zero-filled array."""

    @abstractmethod
    def random_key(self, seed: int) -> Any:
        """Create a backend-specific random key or generator state."""

    @abstractmethod
    def jit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a compiled version of ``fn`` when supported."""

    @abstractmethod
    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a gradient function for ``fn`` when supported."""

    @abstractmethod
    def device_put(self, value: Any) -> Any:
        """Place ``value`` on the backend's default compute device."""
