"""Minimal AlgoRL usage example — batched CW continual learning.

Run from the project root::

    python example.py

Install the package and JAX backend first::

    pip install -e ".[jax]"
"""

from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.envs import resolve_env
from MTCWorldMJX import CWConfig

NUM_ENVS = 32


def main() -> None:
    jax_env = make_batched_cw_train_env(
        "CW10",
        num_envs=NUM_ENVS,
        seed=42,
        config=CWConfig(seed=42, steps_per_task=1_000_000),
    )
    env = resolve_env(jax_env, seed=42)
    agent = arl.EfficientZero(
        env,
        config=EfficientZeroConfig.for_batched(num_envs=NUM_ENVS),
    )
    agent.learn(
        total_timesteps=10_000_000,
        tensorboard_log_dir="runs/cw10_ez_batched_cl",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
