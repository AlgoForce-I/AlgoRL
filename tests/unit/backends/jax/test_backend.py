"""JAX backend tests."""

from algorl.backends.jax.backend import JAXBackend


def test_jax_backend_name() -> None:
    backend = JAXBackend()
    assert backend.name == "jax"


def test_jax_backend_array() -> None:
    backend = JAXBackend()
    value = backend.array([1.0, 2.0])
    assert tuple(value.shape) == (2,)
