from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from MTCWorldMJX import CWConfig

NUM_ENVS = 32
STEPS_PER_TASK = 1_000_000
# Single-task proof of concept: stop after CW10's first task (hammer-v3) to
# check that this learner can solve a Continual World task at all. Raise to
# 10 * STEPS_PER_TASK to run the full sequence.
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
        # Meta-World rewards live in [0, 10]; the (-2, 2) default saturates the
        # reward head so MCTS unrolls never see the 10.0 success payoff.
        reward_support_range=(-10.0, 10.0),
        # CW episodes are 200 steps, so gamma=0.997 (effective horizon ~333)
        # discounts past the episode end. Because time-limit truncation
        # bootstraps, that leaks value across resets into freshly sampled goals.
        # 0.99 (horizon ~100) is also what the Continual World SAC baselines use.
        discount=0.99,
        # Max discounted return at gamma=0.99 over 200 steps of reward 10 is
        # ~866, so this covers the range with headroom while keeping the 51 bins
        # twice as fine as a +/-2000 support.
        value_support_range=(-1000.0, 1000.0),
        # 32 simulations over 16 root candidates leaves only ~2 visits each, so
        # the visit-count policy target is mostly noise and the tree never gets
        # past depth 1. 64 gives ~4 visits per candidate and depth 2-3.
        mcts_simulations=64,
        max_num_considered_actions=16,
        # Exploration hedges: the policy loss is MLE on the single best action,
        # which drives std to its 0.1 floor, and the previous run's entropy sat
        # there. A stronger entropy bonus keeps std up and wider random
        # candidates keep the search from collapsing onto the current mean.
        # Revert these two first if the policy stays too diffuse to grasp.
        entropy_coeff=0.1,
        std_magnification=4.0,
    ).with_schedule_for_run(
        # Schedules span ONE task: the training loop resets plasticity
        # (optimizer state, schedule counters) at every task switch, so
        # warmup / value-mix / TD schedules replay per task.
        STEPS_PER_TASK,
        num_envs=num_envs,
    )

    return config.with_overrides(
        # Freeze the per-task schedule so a longer learn(total_timesteps=...)
        # doesn't restretch it over the full CW10 sequence.
        schedule_horizon="fixed",
        # Step-decay the LR within each task (x0.5 every 300k gradient steps
        # after warmup); the per-task reset restores it to full at each switch.
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
