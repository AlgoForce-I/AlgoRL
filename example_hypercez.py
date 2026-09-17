from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

# MJX/warp allocates outside the JAX pool, on every physics step, and is the
# first thing to fail when XLA's pool grows. XLA retains everything it ever
# claims, so cap it well above its measured peak (~9 GB in-use over 2M steps)
# rather than near the card's size.
configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85, reserve_gb=14.0)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.common.hypercez_retention import HyperCEZRetentionCallback
from MTCWorldMJX import CWConfig

NUM_ENVS = 32
NUM_TASKS = 10
STEPS_PER_TASK = 1_000_000
TOTAL_TIMESTEPS = NUM_TASKS * STEPS_PER_TASK
RUNS_DIR = "/home/algoritmi/data/HyperCEZ_data/runs"
# Fresh run directory: the fix-target run's later boundary checkpoints and
# TensorBoard curves stay untouched and don't overlap with this run's steps.
TENSORBOARD_LOG_DIR = f"{RUNS_DIR}/cw10_hypercez_cl_nullspace"
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
        # Protect earlier tasks by projecting hypernet updates onto their free
        # output directions instead of the β-weighted fix-target regularizer.
        cl_strategy="nullspace",
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
