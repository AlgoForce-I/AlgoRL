"""Config override policy for checkpoint resume."""

from __future__ import annotations

from typing import Any

# Safe to change when resuming (do not reshape stored pytrees).
RESUME_ALLOWLIST: frozenset[str] = frozenset(
    {
        "beta",
        "use_per_task_reg_scaling",
        "reg_scaling_min",
        "reg_scaling_max",
        "warm_start_alpha",
        "scale_hyper_lr",
        "lr_main_to_lr_hyper_ratio",
        "hnet_grad_max_norm",
        "max_grad_norm",
        "retention_log_interval",
        "learning_rate",
        "lr_hyper",
        "lr_warm_up",
        "lr_decay_rate",
        "lr_decay_steps",
        "weight_decay",
        "entropy_coeff",
        "consistency_coeff",
        "reward_loss_coeff",
        "value_loss_coeff",
        "policy_loss_coeff",
        "checkpoint_freq",
        "checkpoint_dir",
        "checkpoint_at_task_boundary",
        "checkpoint_keep_last",
        "autosave_best",
        "autosave_best_metric",
        "autosave_best_window",
        "autosave_best_min_step",
        "memory_log_interval",
        "tensorboard_log_dir",
        "no_look_ahead",
        "dt_scale",
        "use_sgd_change",
        "plastic_prev_tembs",
        "snapshot_shared_per_task",
        "reanalyze_ratio",
        "reanalyze_update_interval",
        "self_play_update_interval",
        "sync_self_play_before_rollout",
        "mcts_simulations",
        "gradient_steps_per_rollout",
        "burst_compile_steps",
        "train_freq",
        "learning_starts",
        "priority_prob_alpha",
        "priority_prob_beta",
        "use_priority",
        "change_temperature",
        "clip_inference_values",
        "std_magnification",
        "alpha_max",
        "frozen_base_weights",
    }
)

# Must match the checkpoint (shape / graph identity).
RESUME_STRUCTURAL: frozenset[str] = frozenset(
    {
        "hnet_arch",
        "hnet_type",
        "hnet_components",
        "emb_size",
        "chunk_dim",
        "cemb_size",
        "num_tasks",
        "hidden_shape",
        "rep_net_shape",
        "dyn_shape",
        "model_type",
        "policy_distribution",
        "support_bins",
        "unroll_steps",
        "trajectory_size",
        "buffer_capacity",
        "search_batch_size",
        "n_stack",
        "num_blocks",
        "backend",
    }
)


def split_resume_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (allowlisted overrides, rejected keys)."""
    allowed: dict[str, Any] = {}
    rejected: list[str] = []
    for key, value in overrides.items():
        if key in RESUME_STRUCTURAL:
            rejected.append(key)
        elif key in RESUME_ALLOWLIST:
            allowed[key] = value
        else:
            rejected.append(key)
    return allowed, rejected
