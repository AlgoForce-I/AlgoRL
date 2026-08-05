from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.common.hypercez_retention import HyperCEZRetentionCallback
from MTCWorldMJX import CWConfig

NUM_ENVS = 32
NUM_TASKS = 10  # CW10
STEPS_PER_TASK = 1_000_000
TOTAL_TIMESTEPS = NUM_TASKS * STEPS_PER_TASK
TENSORBOARD_LOG_DIR = "runs/cw10_hypercez_cl"


def for_cw10_hypercez_batched(
    num_envs: int = 32,
    **overrides: object,
) -> arl.HyperCEZConfig:
    """CW10 HyperCEZ config: EZ knobs + retention-first CL settings.

    EfficientZero fields match ``example.py``. Continual-learning fields follow
    HyperCEZ-master ``main.py`` strength (β=0.5, no plastic prev embeddings)
    while keeping a chunked hypernet for throughput. Per-task LR reset and
    unscaled ``lr_hyper`` are handled in the learner.
    """
    config = arl.HyperCEZConfig.for_batched(num_envs=num_envs).with_overrides(
        # --- EfficientZero (same as example.py) ---
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
        TOTAL_TIMESTEPS,
        num_envs=num_envs,
    )

    return config.with_overrides(
        # --- EfficientZero schedule: applied *per task* by HyperCEZLearner ---
        schedule_horizon="fixed",
        lr_decay_steps=300_000,
        lr_decay_rate=0.5,
        # --- HyperCEZDelta / continual learning ---
        num_tasks=NUM_TASKS,
        steps_per_task=STEPS_PER_TASK,
        hnet_type="chunked",
        hnet_arch=(20, 20),
        chunk_dim=2000,
        cemb_size=20,
        emb_size=10,
        cemb_init_std=1.0,
        emb_init_std=1.0,
        lr_hyper=3e-4,
        scale_hyper_lr=False,
        # Match reference main.py CW10 β (was 0.005 chunked smoke default).
        beta=0.5,
        alpha_max=0.2,
        alpha_init=1e-3,
        no_look_ahead=False,
        dt_scale=1.0,
        use_sgd_change=False,
        plastic_prev_tembs=False,
        warm_start_alpha=True,
        snapshot_shared_per_task=True,
        use_per_task_reg_scaling=False,
        hnet_grad_max_norm=5.0,
        retention_log_interval=500,
        **overrides,
    )


def main() -> None:
    env = make_batched_cw_train_env(
        "CW10",
        num_envs=NUM_ENVS,
        seed=42,
        config=CWConfig(seed=42, steps_per_task=STEPS_PER_TASK),
    )
    config = for_cw10_hypercez_batched(num_envs=NUM_ENVS)
    agent = arl.HyperCEZ(env, config=config)
    # TrainingLoop syncs learner.task_id from env.current_task_index and calls
    # on_task_boundary (snapshot reg targets + shared leaves, warm-start α,
    # reset per-task LR) at each CW task switch; the replay buffer is cleared.
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
