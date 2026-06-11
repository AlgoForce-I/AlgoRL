"""PyTorch backend implementation (stub).

Implement this class when adding the PyTorch backend. It should mirror ``JAXBackend`` and
provide the same ``Backend`` interface: array creation, device placement, compile, and grad.
"""

from __future__ import annotations

from typing import Any, Callable


class TorchBackend:
    """PyTorch backend placeholder."""

    name = "torch"

    def __init__(self) -> None:
        # Implement: initialize default torch device and dtype settings.
        raise NotImplementedError(
            "PyTorch backend is not implemented yet. Use backend='jax'."
        )

    def array(self, value: Any) -> Any:
        # Implement: convert Python / NumPy input to a torch.Tensor.
        raise NotImplementedError

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        # Implement: return a zero-filled torch.Tensor.
        raise NotImplementedError

    def random_key(self, seed: int) -> Any:
        # Implement: seed and return torch RNG state for reproducibility.
        raise NotImplementedError

    def jit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Implement: optionally wrap ``fn`` with torch.compile or equivalent.
        raise NotImplementedError

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Implement: return a gradient function using torch autograd.
        raise NotImplementedError

    def device_put(self, value: Any) -> Any:
        # Implement: move ``value`` onto the default torch device.
        raise NotImplementedError
