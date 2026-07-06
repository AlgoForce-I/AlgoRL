"""EfficientZero target computation tests."""

import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.buffers.efficientzero import (
    bootstrapped_values,
    extended_target_window,
    gae_values,
    mix_value_targets,
    prepare_bootstrapped_batch_values,
    prepare_gae_batch_values,
)


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


def test_prepare_bootstrapped_values_uses_full_td_horizon_at_every_position() -> None:
    """Regression: HyperCEZ bootstraps ``td_steps`` beyond the unroll window."""
    config = EfficientZeroConfig(
        unroll_steps=2,
        td_steps=3,
        discount=0.9,
        value_target="mixed",  # disables the off-policy td shrink, as in HyperCEZ
        model_value_target="bootstrapped",
    )
    ext_window = extended_target_window(config)
    assert ext_window == 2 + 1 + 3

    rewards = np.arange(1, ext_window + 1, dtype=np.float32).reshape(1, -1)
    observations = np.zeros((1, ext_window, 4), dtype=np.float32)
    value = 7.0

    targets = prepare_bootstrapped_batch_values(
        observations,
        rewards,
        np.asarray([0], dtype=np.int32),
        valid_lengths=np.asarray([ext_window], dtype=np.int32),
        total_transitions=1,
        config=config,
        infer_values=lambda obs: np.full((obs.shape[0],), value, dtype=np.float32),
    )

    assert targets.shape == (1, 3)
    discount = config.discount
    for position in range(3):
        expected = sum(
            discount**step * rewards[0, position + step] for step in range(3)
        ) + discount**3 * value
        assert targets[0, position] == pytest.approx(expected, rel=1e-5)


def test_prepare_bootstrapped_values_shrinks_horizon_near_trajectory_end() -> None:
    config = EfficientZeroConfig(
        unroll_steps=2,
        td_steps=3,
        discount=0.9,
        value_target="mixed",
        model_value_target="bootstrapped",
    )
    ext_window = extended_target_window(config)
    rewards = np.ones((1, ext_window), dtype=np.float32)
    rewards[0, 2:] = 0.0  # only 2 core transitions hold real rewards
    observations = np.zeros((1, ext_window, 4), dtype=np.float32)
    value = 5.0

    targets = prepare_bootstrapped_batch_values(
        observations,
        rewards,
        np.asarray([0], dtype=np.int32),
        valid_lengths=np.asarray([2], dtype=np.int32),
        total_transitions=1,
        config=config,
        infer_values=lambda obs: np.full((obs.shape[0],), value, dtype=np.float32),
    )

    discount = config.discount
    # position 0: td shrinks to 2 -> r0 + g*r1 + g^2 * v
    assert targets[0, 0] == pytest.approx(1.0 + discount + discount**2 * value, rel=1e-5)
    # position 1: td shrinks to 1 -> r1 + g * v
    assert targets[0, 1] == pytest.approx(1.0 + discount * value, rel=1e-5)
    # position 2 == valid length: bootstrap masked, no rewards left -> 0
    assert targets[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_prepare_gae_values_matches_constant_case() -> None:
    config = EfficientZeroConfig(
        unroll_steps=2,
        td_lambda=0.95,
        gae_max_steps=15,
        discount=0.9,
        auto_td_steps=30_000,
        model_value_target="GAE",
    )
    ext_window = extended_target_window(config)
    reward = 0.5
    value = 2.0
    rewards = np.full((1, ext_window), reward, dtype=np.float32)
    observations = np.zeros((1, ext_window, 4), dtype=np.float32)

    targets = prepare_gae_batch_values(
        observations,
        rewards,
        np.asarray([0], dtype=np.int32),
        valid_lengths=np.asarray([100], dtype=np.int32),
        bootstrap_limits=np.asarray([ext_window - 1], dtype=np.int32),
        total_transitions=0,
        config=config,
        infer_values=lambda obs: np.full((obs.shape[0],), value, dtype=np.float32),
    )

    discount = config.discount
    td_lambda = config.td_lambda
    span = ext_window - 1
    delta = reward + discount * value - value
    for position in range(3):
        advantage = sum(
            (discount * td_lambda) ** k * delta for k in range(span - position)
        )
        assert targets[0, position] == pytest.approx(advantage + value, rel=1e-4)
