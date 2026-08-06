from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.common.hypercez_retention import HyperCEZRetentionCallback
from example_hypercez import (
    NUM_ENVS,
    STEPS_PER_TASK,
    TOTAL_TIMESTEPS,
    for_cw10_hypercez_batched,
)
from MTCWorldMJX import CWConfig

TENSORBOARD_LOG_DIR = "runs/cw10_hypercez_cl_plastic_w0"


def for_cw10_hypercez_plastic_batched(
    num_envs: int = 32,
    **overrides: object,
) -> arl.HyperCEZConfig:
    """Same as ``example_hypercez`` but with a slowly plastic W0 backbone.

    ``frozen_base_weights=False`` lets the generated base move at
    ``lr_W0 = lr_hyper / lr_main_to_lr_hyper_ratio`` while hypernets still
    learn task-specific residuals.
    """
    return for_cw10_hypercez_batched(num_envs=num_envs).with_overrides(
        frozen_base_weights=False,
        lr_main_to_lr_hyper_ratio=50.0,
        **overrides,
    )


def main() -> None:
    env = make_batched_cw_train_env(
        "CW10",
        num_envs=NUM_ENVS,
        seed=42,
        config=CWConfig(seed=42, steps_per_task=STEPS_PER_TASK),
    )
    config = for_cw10_hypercez_plastic_batched(num_envs=NUM_ENVS)
    agent = arl.HyperCEZ(env, config=config)
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        progress_bar=True,
        callbacks=HyperCEZRetentionCallback(
            agent.learner,
            eval_every_steps=STEPS_PER_TASK // 10,
        ),
    )


if __name__ == "__main__":
    main()
