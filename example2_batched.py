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
from algorl.backends.jax.envs import make_gymnasium_vector_env
from algorl.envs import resolve_env

NUM_ENVS = 32
TOTAL_TIMESTEPS = 10_000_000
# Orange lrdecay2m replay (Jul 17 2026): post-4417e35 code, manual 2M-step LR cliff
# at default EZ-V2 rate 0.1, fixed 10M schedule. Success check by ~500K env steps:
# median return should climb toward ~4K+, not plateau near ~2K.
LR_DECAY_STEPS = 2_000_000
LR_DECAY_RATE = 0.1
TENSORBOARD_LOG_DIR = "runs/half_cheetah_ez_batched_lrdecay2m_replay_nosync"


def for_half_cheetah_batched(
    num_envs: int = 32,
    **overrides: object,
) -> arl.EfficientZeroConfig:
    """Batched Gym HalfCheetah preset (wider value support than DMC default)."""
    config = arl.EfficientZeroConfig.for_batched(num_envs=num_envs).with_overrides(
        value_support_range=(-5000.0, 5000.0),
        clip_inference_values=False,
        reward_support_range=(-10.0, 10.0),
        use_bn=True,
        lr_warm_up=0.01,
    ).with_schedule_for_run(
        TOTAL_TIMESTEPS,
        num_envs=num_envs,
    )

    return config.with_overrides(
        schedule_horizon="fixed",
        total_training_steps=config.total_training_steps,
        start_use_mix_training_steps=config.start_use_mix_training_steps,
        auto_td_steps=config.auto_td_steps,
        mixed_value_threshold=config.mixed_value_threshold,
        lr_decay_steps=LR_DECAY_STEPS,
        lr_decay_rate=LR_DECAY_RATE,
        **overrides,
    )


def main() -> None:
    jax_env = make_gymnasium_vector_env(
        lambda: gym.make("HalfCheetah-v5"),
        NUM_ENVS,
        vector_cls=gym.vector.AsyncVectorEnv,
        seed=42,
    )
    env = resolve_env(jax_env, seed=42)
    agent = arl.EfficientZero(
        env,
        config=for_half_cheetah_batched(num_envs=NUM_ENVS),
    )
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
