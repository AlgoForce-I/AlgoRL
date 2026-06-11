"""PyTorch backend stub tests."""

import pytest

from algorl.backends.torch.backend import TorchBackend


def test_torch_backend_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="PyTorch backend"):
        TorchBackend()
