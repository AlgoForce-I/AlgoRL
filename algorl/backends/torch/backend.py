"""PyTorch backend implementation (stub).

Implement this class when adding the PyTorch backend. It should mirror ``JAXBackend`` and
provide the same ``Backend`` interface: array creation, device placement, compile, and grad.
Neural-network modules and optimizers remain in ``backends/torch/`` component packages.
"""

from __future__ import annotations

from typing import Any, Callable

from algorl.core.backend import Backend


class TorchBackend(Backend):
    """PyTorch backend placeholder."""

    name = "torch"

    @property
    def supports_jit(self) -> bool:
        return False

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
        return fn

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError

    def device_put(self, value: Any) -> Any:
        raise NotImplementedError
