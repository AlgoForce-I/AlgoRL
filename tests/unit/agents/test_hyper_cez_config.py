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
    assert config.hnet_arch == (100, 100)
    assert config.hnet_type == "unchunked"
    assert config.chunk_dim == 2000
    assert config.cemb_size == 20
    assert config.cemb_init_std == 1.0
    assert config.emb_size == 10
    assert config.num_tasks == 10
    assert config.lr_hyper == 3e-4
    assert config.beta == 1.0
    assert config.alpha_max == 0.2
    assert config.alpha_init == 1e-3
    assert config.no_look_ahead is False
    assert config.use_sgd_change is False
    assert config.plastic_prev_tembs is False
    assert config.ewc_weight_importance is False
    assert config.scale_hyper_lr is False
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
    config = HyperCEZConfig.for_batched(num_envs=8, beta=0.5, num_tasks=5)
    assert isinstance(config, HyperCEZConfig)
    assert config.search_batch_size == 8
    assert config.beta == 0.5
    assert config.num_tasks == 5
    assert config.batch_size == EfficientZeroConfig.for_batched(num_envs=8).batch_size


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
