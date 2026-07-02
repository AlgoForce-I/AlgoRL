"""EfficientZero target computation tests."""

import numpy as np
import pytest

from algorl.buffers.efficientzero import bootstrapped_values, gae_values, mix_value_targets


def test_bootstrapped_values_matches_n_step_return() -> None:
    rewards = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    targets = bootstrapped_values(rewards, values, discount=0.9, td_steps=2)
    expected_first = 1.0 + 0.9 * 1.0 + (0.9**2) * 0.5
    assert targets[0] == pytest.approx(expected_first)


def test_mix_value_targets_blends_search_and_bootstrapped() -> None:
    bootstrapped = np.array([1.0, 2.0], dtype=np.float32)
    search = np.array([10.0, 20.0], dtype=np.float32)
    mask = np.array([1.0, 0.0], dtype=np.float32)
    mixed = mix_value_targets(bootstrapped, search, use_search_mask=mask)
    assert mixed[0] == pytest.approx(1.0)
    assert mixed[1] == pytest.approx(20.0)
