"""PyTorch backend implementation (stub)."""

from __future__ import annotations

from typing import Any, Callable


class TorchBackend:
    """PyTorch backend placeholder."""

    name = "torch"

    def __init__(self) -> None:
        raise NotImplementedError(
            "PyTorch backend is not implemented yet. Use backend='jax'."
        )

    def array(self, value: Any) -> Any:
        raise NotImplementedError

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        raise NotImplementedError

    def random_key(self, seed: int) -> Any:
        raise NotImplementedError

    def jit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError

    def device_put(self, value: Any) -> Any:
        raise NotImplementedError
