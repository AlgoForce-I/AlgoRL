from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85, reserve_gb=14.0)

import jax

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
TENSORBOARD_LOG_DIR = f"{RUNS_DIR}/cw10_hypercez_fix target"
CHECKPOINT_DIR = f"{TENSORBOARD_LOG_DIR}/checkpoints"
RESUME_FROM: str | None = None


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
        eval_period=200_000,
        eval_episodes=20,
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
