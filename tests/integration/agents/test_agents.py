"""Agent integration tests."""

import algorl as arl


def test_public_exports() -> None:
    assert hasattr(arl, "EfficientZero")
    assert hasattr(arl, "DreamerV3")
    assert arl.__version__ == "0.0.1"


def test_efficient_zero_construction(dummy_env: object) -> None:
    agent = arl.EfficientZero(dummy_env, backend="jax")
    assert agent.backend.name == "jax"
    assert agent.world_model is not None
    assert agent.planner is not None
    assert agent.learner is not None
