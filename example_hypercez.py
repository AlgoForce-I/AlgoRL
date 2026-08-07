from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.common.hypercez_retention import HyperCEZRetentionCallback
from MTCWorldMJX import CWConfig

NUM_ENVS = 32
NUM_TASKS = 10
STEPS_PER_TASK = 1_000_000
TOTAL_TIMESTEPS = NUM_TASKS * STEPS_PER_TASK
TENSORBOARD_LOG_DIR = "runs/cw10_hypercez_cl_unchuncked"
CHECKPOINT_DIR = f"{TENSORBOARD_LOG_DIR}/checkpoints"
# Set to e.g. f"{CHECKPOINT_DIR}/boundary_task_0" to continue after task 0.
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
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_freq=STEPS_PER_TASK,
        autosave_best=True,
        autosave_best_window=20,
    )
    agent = arl.HyperCEZ(env, config=config)
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        resume_from=RESUME_FROM,
        progress_bar=True,
        callbacks=HyperCEZRetentionCallback(
            agent.learner,
            eval_every_steps=STEPS_PER_TASK // 10,
        ),
    )


if __name__ == "__main__":
    main()
