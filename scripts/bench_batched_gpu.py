"""GPU benchmark for the batched EfficientZero pipeline (temporary script)."""

from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import time

import algorl as arl
import gymnasium as gym
import jax
from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.envs import make_gymnasium_vector_env
from algorl.envs import resolve_env

NUM_ENVS = 32


def main() -> None:
    print(f"jax devices: {jax.local_devices()}")
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
    print(f"resolved reanalyze search width: {agent.learner._reanalyze_search_width}")

    start = time.perf_counter()
    # ~7,700 steps: warmup (buffer commits at ~6,400) plus several 320-step bursts.
    agent.learn(total_timesteps=7_680, progress_bar=False)
    elapsed = time.perf_counter() - start

    stats = jax.local_devices()[0].memory_stats() or {}
    in_use = stats.get("bytes_in_use", 0) / 1024**3
    peak = stats.get("peak_bytes_in_use", 0) / 1024**3
    print(f"train steps completed: {agent.learner._train_steps}")
    print(f"elapsed: {elapsed:.1f}s  steps/s: {7_680 / elapsed:.1f}")
    print(f"gpu memory in use: {in_use:.2f} GiB  peak: {peak:.2f} GiB")


if __name__ == "__main__":
    main()
