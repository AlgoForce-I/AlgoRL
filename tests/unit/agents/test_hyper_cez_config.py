"""HyperCEZ config and composition wiring tests."""

from __future__ import annotations

from algorl.agents import compositions
from algorl.agents.configs import (
    DEFAULT_HYPERCEZ_HNET_COMPONENTS,
    EfficientZeroConfig,
    HyperCEZConfig,
)
from algorl.agents.search.hyper_cez import HyperCEZ
from algorl.backends.jax import learners as jax_learners
from algorl.backends.jax import world_models as jax_world_models


def test_hypercez_config_defaults() -> None:
    config = HyperCEZConfig()
    assert config.hnet_components == DEFAULT_HYPERCEZ_HNET_COMPONENTS
    assert config.hnet_arch == (128, 128)
    assert config.hnet_type == "unchunked"
    assert config.chunk_dim == 2000
    assert config.cemb_size == 20
    assert config.cemb_init_std == 1.0
    assert config.emb_size == 10
    assert config.num_tasks == 10
    assert config.steps_per_task == 1_000_000
    assert config.lr_hyper == 3e-4
    assert config.beta == 0.5
    assert config.alpha_max == 1.0
    assert config.alpha_init == 2.0
    assert config.head_init_std == 0.0
    assert config.use_bn is True
    assert config.discount == 0.99
    assert config.mcts_simulations == 64
    assert config.entropy_coeff == 0.1
    assert config.std_magnification == 4.0
    assert config.schedule_horizon == "fixed"
    assert config.lr_decay_rate == 0.5
    assert config.no_look_ahead is False
    assert config.use_sgd_change is False
    assert config.plastic_prev_tembs is False
    assert config.ewc_weight_importance is False
    assert config.scale_hyper_lr is False
    assert config.frozen_base_weights is True
    assert config.lr_main_to_lr_hyper_ratio == 50.0
    assert config.warm_start_alpha is True
    assert config.snapshot_shared_per_task is True
    assert config.use_per_task_reg_scaling is False


def test_hypercez_config_chunked_override() -> None:
    config = HyperCEZConfig().with_overrides(
        hnet_type="chunked",
        chunk_dim=512,
        cemb_size=16,
        hnet_arch=(20, 20),
    )
    assert config.hnet_type == "chunked"
    assert config.chunk_dim == 512
    assert config.cemb_size == 16
    assert config.hnet_arch == (20, 20)


def test_hypercez_config_inherits_ez_presets() -> None:
    config = HyperCEZConfig.for_batched(num_envs=8, beta=0.25, num_tasks=5)
    assert isinstance(config, HyperCEZConfig)
    assert config.search_batch_size == 8
    assert config.beta == 0.25
    assert config.num_tasks == 5
    assert config.use_bn is True
    assert config.discount == 0.99
    assert config.mcts_simulations == 64
    assert config.schedule_horizon == "fixed"
    assert config.lr_decay_steps == 300_000
    assert config.batch_size == EfficientZeroConfig.for_batched(num_envs=8).batch_size


def test_hypercez_for_batched_uses_per_task_schedule() -> None:
    config = HyperCEZConfig.for_batched(num_envs=8, steps_per_task=100_000)
    # Mix/TD horizons come from the per-task budget; lr decay stays fixed.
    assert config.steps_per_task == 100_000
    assert config.total_training_steps > 0
    assert config.lr_decay_steps == 300_000
    assert config.lr_decay_rate == 0.5


def test_hypercez_config_with_overrides() -> None:
    config = HyperCEZConfig.for_sequential().with_overrides(
        emb_size=16,
        hnet_arch=(64, 64),
        alpha_max=0.3,
    )
    assert isinstance(config, HyperCEZConfig)
    assert config.emb_size == 16
    assert config.hnet_arch == (64, 64)
    assert config.alpha_max == 0.3
    assert config.use_bn is True


def test_hypercez_composition_slots() -> None:
    composition = compositions.registry.create("hyper_cez")
    assert composition.components == {
        "world_model": "hyper_cez",
        "planner": "mcts",
        "learner": "hyper_cez",
        "replay_buffer": "efficient_zero",
    }


def test_hypercez_backend_kinds_registration() -> None:
    assert not jax_world_models.registry.is_stub("hyper_cez")
    assert not jax_learners.registry.is_stub("hyper_cez")


def test_hypercez_agent_class_wiring() -> None:
    assert HyperCEZ.composition_name == "hyper_cez"
    assert HyperCEZ.config_class is HyperCEZConfig
