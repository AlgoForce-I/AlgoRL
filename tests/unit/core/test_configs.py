"""Agent config tests."""

import pytest

from algorl.agents.configs import EfficientZeroConfig


def test_config_with_overrides() -> None:
    config = EfficientZeroConfig().with_overrides(mcts_simulations=25, seed=7)
    assert config.mcts_simulations == 25
    assert config.seed == 7
    assert config.backend == "jax"


def test_config_rejects_unknown_overrides() -> None:
    with pytest.raises(TypeError, match="Unknown config field"):
        EfficientZeroConfig().with_overrides(unknown_field=1)
