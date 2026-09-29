from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from MTCWorldMJX import CWConfig

NUM_ENVS = 32
STEPS_PER_TASK = 1_000_000
TOTAL_TIMESTEPS = STEPS_PER_TASK
RUNS_DIR = "/home/algoritmi/data/HyperCEZ_data/runs"
# Single-task control for HyperCEZ's stick-pull (task 4 of CW10): same config,
# seed, budget and eval protocol, trained from scratch without continual learning.
TASK_NAME = "stick-pull-v3"
TENSORBOARD_LOG_DIR = f"{RUNS_DIR}/cw10_ez_stick_pull"
# Match example_hypercez.py so the two eval curves are directly comparable.
EVAL_PERIOD = 200_000
EVAL_EPISODES = 20


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

    # Set after the schedule so mixed_value_threshold (0.2 x capacity) stays
    # 20k; top_transitions caps the sampled window and must match capacity.
    return config.with_overrides(
        schedule_horizon="fixed",
        lr_decay_steps=300_000,
        lr_decay_rate=0.5,
        buffer_capacity=500_000,
        top_transitions=500_000.0,
        **overrides,
    )


def main() -> None:
    env = make_batched_cw_train_env(
        "CW10",
        num_envs=NUM_ENVS,
        seed=42,
        config=CWConfig(seed=42, steps_per_task=STEPS_PER_TASK),
        task_name=TASK_NAME,
    )
    agent = arl.EfficientZero(
        env,
        config=for_cw10_batched(num_envs=NUM_ENVS),
    )
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_period=EVAL_PERIOD,
        eval_episodes=EVAL_EPISODES,
        tensorboard_log_dir=TENSORBOARD_LOG_DIR,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
