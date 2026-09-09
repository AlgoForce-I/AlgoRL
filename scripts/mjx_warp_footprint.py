"""Report the MJX/warp device footprint that must stay outside the JAX pool."""

from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import jax
import jax.numpy as jnp
import numpy as np

from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.core.evaluation import EvalTask, _make_eval_env
from MTCWorldMJX import CWConfig

NUM_ENVS = 32


def report(label: str) -> None:
    import warp as wp

    device = wp.get_device("cuda:0")
    total = device.total_memory / 1024.0**3
    free = device.free_memory / 1024.0**3
    pool = wp.get_mempool_used_mem_high(device) / 1024.0**3
    jax_stats = jax.local_devices()[0].memory_stats() or {}
    jax_used = float(jax_stats.get("bytes_in_use", 0)) / 1024.0**3
    print(
        f"{label:<34} used={total - free:6.3f}  free={free:6.3f}  "
        f"warp_pool_high={pool:6.3f}  jax_in_use={jax_used:6.3f} GB"
    )


def main() -> None:
    config = CWConfig(seed=42, steps_per_task=1_000_000)
    env = make_batched_cw_train_env(
        "CW10", num_envs=NUM_ENVS, seed=42, config=config
    )
    report("train env built")

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    for _ in range(5):
        state = env.step(state, jnp.zeros((NUM_ENVS, 4), dtype=jnp.float32))
    report("train env stepped")

    bench = env.benchmark
    eval_envs = []
    for task_index, name in enumerate(bench.task_names):
        task = EvalTask(
            name=name,
            task_index=task_index,
            kind="cw",
            benchmark="CW10",
            cw_config=bench.config,
            max_steps=5,
        )
        eval_env = _make_eval_env(task, num_envs=3, seed=0, bench=bench)
        obs = eval_env.reset(seed=0)
        for _ in range(3):
            obs, *_ = eval_env.step(np.zeros((3, 4), dtype=np.float32))
        eval_env.release()
        eval_envs.append(eval_env)
        report(f"+ eval env {task_index} cached")


if __name__ == "__main__":
    main()
