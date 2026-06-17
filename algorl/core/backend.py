"""Backend abstract base class for framework isolation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Backend(ABC):
    """Minimal array and execution helpers for one ML framework.

    Backends intentionally do **not** own neural-network modules, optimizers, or
    checkpoint formats. Those live in backend-specific component implementations
    under ``backends/<name>/``. This class only exposes the small set of primitives
    that core code may need without leaking framework types through public APIs.
    """

    name: str

    @property
    def supports_jit(self) -> bool:
        """Whether ``jit`` provides real compilation for this backend."""
        return True

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
