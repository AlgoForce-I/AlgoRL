from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

# MJX/warp allocates outside the JAX pool, on every physics step, and is the
# first thing to fail when XLA's pool grows. XLA retains everything it ever
# claims, so cap it well above its measured peak (~9 GB in-use over 2M steps)
# rather than near the card's size.
configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85, reserve_gb=14.0)

import jax

# Reuse compiled executables across restarts and task boundaries. Same programs,
# same results; it only skips recompilation (minutes per boundary / restart).
jax.config.update("jax_compilation_cache_dir", "/home/algoritmi/data/HyperCEZ_data/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.common.hypercez_retention import HyperCEZRetentionCallback
from MTCWorldMJX import CWConfig

NUM_ENVS = 32
NUM_TASKS = 10
STEPS_PER_TASK = 1_000_000
TOTAL_TIMESTEPS = NUM_TASKS * STEPS_PER_TASK
RUNS_DIR = "/home/algoritmi/data/HyperCEZ_data/runs"
# Fresh run directory: the earlier run's later boundary checkpoints and
# TensorBoard curves stay untouched and don't overlap with this run's steps.
# v2 = two-sided λ controller (reg_task_share_floor). The v1 curves in
# cw10_hypercez_cl_balanced are the saturated ones: λ≈900, task share 0.12%.
TENSORBOARD_LOG_DIR = f"{RUNS_DIR}/cw10_hypercez_cl_balanced_v2"
CHECKPOINT_DIR = f"{TENSORBOARD_LOG_DIR}/checkpoints"
# Written when task 0 (hammer) finished; the run continues with task 1
# (push-wall). Task 0 never used the regularizer, so this state is identical
# under either CL strategy. `boundary_task_{k}` is saved after task k ends.
RESUME_FROM: str | None = f"{RUNS_DIR}/cw10_hypercez_cl_unchuncked_fixes/checkpoints/boundary_task_0"


def main() -> None:
    env = make_batched_cw_train_env(
        "CW10",
        num_envs=NUM_ENVS,
        seed=42,
        config=CWConfig(seed=42, steps_per_task=STEPS_PER_TASK),
    )
    config = arl.HyperCEZConfig.for_batched(
        num_envs=NUM_ENVS,
        # Fix-target regularizer, balanced per hypernet component in gradient
        # space. λ tightens while earlier tasks' generated weights are drifting
        # past reg_drift_budget, and never past the point where the new task
        # keeps reg_task_share_floor of its own gradient.
        # cl_strategy="nullspace" swaps in exact null-space projection instead.
        cl_strategy="fix_target",
        reg_balance="gradient",
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_freq=STEPS_PER_TASK,
        autosave_best=True,
        autosave_best_window=20,
    )
    agent = arl.HyperCEZ(env, config=config)
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_period=100_000,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        resume_from=RESUME_FROM,
        progress_bar=True,
        callbacks=HyperCEZRetentionCallback(
            agent.learner,
            retention_every_steps=STEPS_PER_TASK // 10,
        ),
    )


if __name__ == "__main__":
    main()
