"""EfficientZero schedule auto-resolution tests."""

from algorl.agents.configs import EfficientZeroConfig
from algorl.buffers.efficientzero.schedule import (
    estimate_gradient_step_budget,
    resolve_efficient_zero_schedule,
    schedule_ratios,
)


def test_dmc_state_schedule_ratios() -> None:
    config = EfficientZeroConfig.for_dmc_state(model_type="dmc_state")
    ratios = schedule_ratios(config)
    assert ratios["mix_start"] == 0.4
    assert ratios["auto_td"] == 0.6
    assert ratios["mixed_value_buffer"] == 0.2


def test_estimate_gradient_budget_sequential_train_freq_one() -> None:
    config = EfficientZeroConfig.for_dmc_state(
        learning_starts=2_000,
        train_freq=1,
    )
    budget = estimate_gradient_step_budget(10_000_000, config)
    assert budget == 9_998_000


def test_resolve_schedule_scales_with_run_length() -> None:
    config = EfficientZeroConfig.for_dmc_state(
        schedule_horizon="auto",
        learning_starts=2_000,
        train_freq=1,
        buffer_capacity=100_000,
    )
    resolved = resolve_efficient_zero_schedule(config, 10_000_000)

    assert resolved.total_training_steps == 9_998_000
    assert resolved.start_use_mix_training_steps == int(9_998_000 * 0.4)
    assert resolved.auto_td_steps == int(9_998_000 * 0.6)
    assert resolved.mixed_value_threshold == 20_000.0
    assert resolved.lr_decay_steps == int(300_000 * 9_998_000 / 100_000)


def test_resolve_schedule_keeps_lr_decay_near_default_for_reference_length() -> None:
    config = EfficientZeroConfig.for_dmc_state(
        schedule_horizon="auto",
        learning_starts=2_000,
        train_freq=1,
    )
    resolved = resolve_efficient_zero_schedule(config, 100_000)
    assert resolved.total_training_steps == 98_000
    assert resolved.lr_decay_steps == int(300_000 * 98_000 / 100_000)


def test_resolve_schedule_scales_lr_decay_for_batched_run() -> None:
    config = EfficientZeroConfig.for_batched(num_envs=32).with_schedule_for_run(
        10_000_000,
        num_envs=32,
    )
    grad_budget = config.total_training_steps
    assert config.lr_decay_steps == int(300_000 * grad_budget / 100_000)


def test_fixed_schedule_horizon_ignores_run_length() -> None:
    config = EfficientZeroConfig.for_dmc_state(
        schedule_horizon="fixed",
        total_training_steps=100_000,
        start_use_mix_training_steps=40_000,
        auto_td_steps=60_000,
        mixed_value_threshold=20_000.0,
    )
    resolved = resolve_efficient_zero_schedule(config, 10_000_000)
    assert resolved.total_training_steps == 100_000
    assert resolved.start_use_mix_training_steps == 40_000
    assert resolved.lr_decay_steps == 300_000
