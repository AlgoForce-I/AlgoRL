"""JAX backend implementation."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


class JAXBackend:
    """JAX/Flax backend."""

    name = "jax"

    def array(self, value: Any) -> Any:
        return jnp.asarray(value)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return jnp.zeros(shape, dtype=dtype)

    def random_key(self, seed: int) -> Any:
        return jax.random.PRNGKey(seed)

    def jit(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return jax.jit(fn)

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return jax.grad(fn)

    def device_put(self, value: Any) -> Any:
        return jax.device_put(value)
