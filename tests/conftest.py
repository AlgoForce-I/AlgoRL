"""Pytest configuration."""

import gymnasium as gym
import pytest


@pytest.fixture
def cartpole_env() -> gym.Env:
    """Standard Gymnasium environment for agent tests."""
    return gym.make("CartPole-v1")
