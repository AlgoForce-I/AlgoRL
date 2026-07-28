from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from MTCWorldMJX import CWConfig

NUM_ENVS = 32
STEPS_PER_TASK = 1_000_000
TOTAL_TIMESTEPS = STEPS_PER_TASK
TENSORBOARD_LOG_DIR = "runs/cw10_ez_hammer_poc"


def for_cw10_batched(
    num_envs: int = 32,
    **overrides: object,
) -> arl.EfficientZeroConfig:
    config = arl.EfficientZeroConfig.for_batched(num_envs=num_envs).with_overrides(
        use_bn=True,
        lr_warm_up=0.01,
        clip_inference_values=True,
        change_temperature=False,
        reward_support_range=(-10.0, 10.0),
        discount=0.99,
        value_support_range=(-1000.0, 1000.0),
        mcts_simulations=64,
        max_num_considered_actions=16,
        entropy_coeff=0.1,
        std_magnification=4.0,
    ).with_schedule_for_run(
        STEPS_PER_TASK,
        num_envs=num_envs,
    )

    return config.with_overrides(
        schedule_horizon="fixed",
        lr_decay_steps=300_000,
        lr_decay_rate=0.5,
        **overrides,
    )


def main() -> None:
    env = make_batched_cw_train_env(
        "CW10",
        num_envs=NUM_ENVS,
        seed=42,
        config=CWConfig(seed=42, steps_per_task=STEPS_PER_TASK),
    )
    agent = arl.EfficientZero(
        env,
        config=for_cw10_batched(num_envs=NUM_ENVS),
    )
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
