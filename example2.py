"""Minimal AlgoRL usage example — HalfCheetah (single Gymnasium env).

Run from the project root::

    python example2.py

Install the package and JAX backend first::

    pip install -e ".[jax]"
"""

from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
import gymnasium as gym


def main() -> None:
    env = gym.make("HalfCheetah-v5")
    agent = arl.EfficientZero(
        env,
        config=arl.EfficientZeroConfig.for_sequential(),
    )
    agent.learn(
        total_timesteps=10_000_000,
        tensorboard_log_dir="runs/half_cheetah_ez",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
