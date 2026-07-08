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
        config=EfficientZeroConfig.for_batched(num_envs=NUM_ENVS),
    )
    agent.learn(
        total_timesteps=10_000_000,
        tensorboard_log_dir="runs/half_cheetah_ez_batched",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
