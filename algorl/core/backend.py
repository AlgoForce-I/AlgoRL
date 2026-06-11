"""Backend protocol for framework isolation."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Minimal interface implemented by each ML backend."""

    name: str

    def array(self, value: Any) -> Any:
        """Convert a Python or NumPy value to a backend array."""

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create a zero-filled array."""

    def random_key(self, seed: int) -> Any:
        """Create a backend-specific random key or generator state."""

    def jit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a compiled version of ``fn`` when supported."""

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a gradient function for ``fn`` when supported."""

    def device_put(self, value: Any) -> Any:
        """Place ``value`` on the backend's default compute device."""
