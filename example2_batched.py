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
from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.envs import make_gymnasium_vector_env
from algorl.envs import resolve_env

NUM_ENVS = 32


def half_cheetah_batched_config(*, num_envs: int) -> EfficientZeroConfig:
    """``for_batched`` with Gym HalfCheetah overrides (DMC preset stays intact)."""
    return EfficientZeroConfig.for_batched(num_envs=num_envs).with_overrides(
        # HalfCheetah returns reach thousands; DMC ±299 support caps value learning.
        value_support_range=(-5000.0, 5000.0),
        # Random-policy episodes are strongly negative; do not floor MCTS values at 0.
        clip_inference_values=False,
        # Per-step rewards can exceed the DMC default once the agent runs forward.
        reward_support_range=(-10.0, 10.0),
        use_bn=True,
        lr_warm_up=0.01,
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
        config=half_cheetah_batched_config(num_envs=NUM_ENVS),
    )
    agent.learn(
        total_timesteps=10_000_000,
        tensorboard_log_dir="runs/half_cheetah_ez_batched_fixcheck_from_cursor",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
