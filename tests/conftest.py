"""Pytest configuration."""

import pytest


@pytest.fixture
def dummy_env() -> object:
    """Minimal environment stub for agent construction tests."""

    class _DummyEnv:
        def reset(self, **kwargs):
            return None, {}

        def step(self, action):
            return None, 0.0, False, False, {}

    return _DummyEnv()
