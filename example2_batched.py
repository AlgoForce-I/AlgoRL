"""Minimal AlgoRL usage example — batched HalfCheetah (parallel Gymnasium envs).

Run from the project root::

    python example2_batched.py

Install the package and JAX backend first::

    pip install -e ".[jax]"
"""

from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
import gymnasium as gym
from algorl.buffers.efficientzero.schedule import (
    estimate_gradient_step_budget,
    schedule_ratios,
)

NUM_ENVS = 32
TOTAL_TIMESTEPS = 10_000_000
# Soft anneal (best guess after lrdecay2m dips): full 3e-4 through the climb,
# then ×0.5 every 2M train steps → ~1.5e-4 @2.1M, ~7.5e-5 @4.1M, ~3.8e-5 @6.1M.
# Avoids the old ×0.1 cliffs to 3e-6/3e-7 that stacked with mix_start / temp drops.
LR_DECAY_STEPS = 2_000_000
LR_DECAY_RATE = 0.5


def for_half_cheetah_batched(
    num_envs: int = 32,
    *,
    total_timesteps: int = TOTAL_TIMESTEPS,
    **overrides: object,
) -> arl.EfficientZeroConfig:
    """Batched Gym HalfCheetah preset (wider value support than DMC default)."""
    config = arl.EfficientZeroConfig.for_batched(num_envs=num_envs).with_overrides(
        value_support_range=(-5000.0, 5000.0),
        clip_inference_values=False,
        reward_support_range=(-10.0, 10.0),
        use_bn=True,
        lr_warm_up=0.01,
        **overrides,
    )
    grad_budget = estimate_gradient_step_budget(
        total_timesteps,
        config,
        num_envs=num_envs,
    )
    ratios = schedule_ratios(config)
    return config.with_overrides(
        schedule_horizon="fixed",
        total_training_steps=grad_budget,
        start_use_mix_training_steps=max(1, int(grad_budget * ratios["mix_start"])),
        auto_td_steps=max(1, int(grad_budget * ratios["auto_td"])),
        mixed_value_threshold=max(
            1.0,
            float(config.buffer_capacity) * ratios["mixed_value_buffer"],
        ),
        lr_decay_steps=LR_DECAY_STEPS,
        lr_decay_rate=LR_DECAY_RATE,
    )


def main() -> None:
    env = gym.make("HalfCheetah-v5")
    agent = arl.EfficientZero(
        env,
        config=for_half_cheetah_batched(num_envs=NUM_ENVS, total_timesteps=TOTAL_TIMESTEPS),
    )
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tensorboard_log_dir="runs/half_cheetah_ez_batched_softlr",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
