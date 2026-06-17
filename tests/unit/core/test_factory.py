"""Core factory tests."""

import pytest

from algorl.core.factory import get_backend


def test_get_backend_jax() -> None:
    backend = get_backend("jax")
    assert backend.name == "jax"
    assert backend.zeros((2, 2)).shape == (2, 2)


def test_get_backend_torch_raises() -> None:
    with pytest.raises(NotImplementedError, match="PyTorch backend"):
        get_backend("torch")


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("unknown")
