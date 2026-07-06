"""Training schedule resolution for EfficientZero (HyperCEZ-aligned ratios)."""

from __future__ import annotations

import math

from algorl.agents.configs import EfficientZeroConfig

# HyperCEZ ``ez_hparams.json`` schedule ratios (gradient-step budget =
# ``training_steps + offline_training_steps``; mixed-value window vs ``buffer_size``).
_HYPERCEZ_SCHEDULE_RATIOS: dict[str, dict[str, float]] = {
    "atari": {
        "mix_start": 30_000 / 100_000,
        "auto_td": 30_000 / 100_000,
        "mixed_value_buffer": 5_000 / 1_000_000,
    },
    "dmc_image": {
        "mix_start": 40_000 / 120_000,
        "auto_td": 30_000 / 120_000,
        "mixed_value_buffer": 20_000 / 200_000,
    },
    "dmc_state": {
        "mix_start": 40_000 / 100_000,
        "auto_td": 60_000 / 100_000,
        "mixed_value_buffer": 20_000 / 100_000,
    },
}


def _schedule_preset_key(config: EfficientZeroConfig) -> str:
    model_type = config.model_type
    if model_type == "auto":
        return "dmc_state"
    if model_type not in _HYPERCEZ_SCHEDULE_RATIOS:
        return "dmc_state"
    return model_type


def schedule_ratios(config: EfficientZeroConfig) -> dict[str, float]:
    """Return effective schedule ratios for ``config`` (preset or explicit overrides)."""
    preset = _HYPERCEZ_SCHEDULE_RATIOS[_schedule_preset_key(config)]
    mix_start = (
        config.schedule_mix_start_fraction
        if config.schedule_mix_start_fraction is not None
        else preset["mix_start"]
    )
    auto_td = (
        config.schedule_auto_td_fraction
        if config.schedule_auto_td_fraction is not None
        else preset["auto_td"]
    )
    mixed_buffer = (
        config.schedule_mixed_value_buffer_fraction
        if config.schedule_mixed_value_buffer_fraction is not None
        else preset["mixed_value_buffer"]
    )
    return {
        "mix_start": mix_start,
        "auto_td": auto_td,
        "mixed_value_buffer": mixed_buffer,
    }


def estimate_gradient_step_budget(
    total_timesteps: int,
    config: EfficientZeroConfig,
    *,
    num_envs: int = 1,
) -> int:
    """Estimate gradient updates over a run (matches sequential / batched training loops)."""
    if total_timesteps <= config.learning_starts:
        return 1

    trainable_env_steps = total_timesteps - config.learning_starts
    train_freq = max(1, config.train_freq)
    capped = config.gradient_steps_per_rollout

    if capped is not None and num_envs > 1:
        chunk_env_steps = max(1, config.jax_rollout_chunk * max(1, num_envs))
        num_chunks = math.ceil(trainable_env_steps / chunk_env_steps)
        per_chunk = max(0, capped)
        return max(1, num_chunks * per_chunk)

    return max(1, trainable_env_steps // train_freq)


def resolve_efficient_zero_schedule(
    config: EfficientZeroConfig,
    total_timesteps: int,
    *,
    num_envs: int = 1,
) -> EfficientZeroConfig:
    """Derive schedule horizons from ``learn(total_timesteps=...)`` when ``schedule_horizon='auto'``."""
    if config.schedule_horizon != "auto":
        return config

    grad_budget = estimate_gradient_step_budget(
        total_timesteps,
        config,
        num_envs=num_envs,
    )
    ratios = schedule_ratios(config)
    mix_start = max(1, int(grad_budget * ratios["mix_start"]))
    auto_td = max(1, int(grad_budget * ratios["auto_td"]))
    mixed_threshold = max(
        1.0,
        float(config.buffer_capacity) * ratios["mixed_value_buffer"],
    )

    return config.with_overrides(
        total_training_steps=grad_budget,
        start_use_mix_training_steps=mix_start,
        auto_td_steps=auto_td,
        mixed_value_threshold=mixed_threshold,
    )
