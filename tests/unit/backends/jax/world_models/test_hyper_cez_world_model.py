"""HyperCEZ world model tests."""

from __future__ import annotations

import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

from algorl.agents.configs import BaseAgentConfig, HyperCEZConfig
from algorl.backends.jax.factory import JAXComponentFactory
from algorl.backends.jax.nn.hypercez.shapes import partition_params
from algorl.backends.jax.planners.efficientzero.planner import EfficientZeroPlanner
from algorl.backends.jax.world_models.efficientzero import EfficientZeroWorldModel
from algorl.backends.jax.world_models.hypercez import (
    HyperCEZWorldModel,
    build_hyper_cez_world_model,
)
from algorl.core.component_context import ComponentContext
from algorl.core.factory import get_backend
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_training_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


@pytest.fixture
def hypercez_world_model(cartpole_training_env: TrainingEnv) -> HyperCEZWorldModel:
    config = HyperCEZConfig.for_dmc_state(
        num_tasks=3,
        emb_size=8,
        hnet_arch=(32, 32),
        head_init_std=1e-3,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    return build_hyper_cez_world_model(context)


def test_hypercez_world_model_registered_as_implemented() -> None:
    factory = JAXComponentFactory()
    assert not factory.is_stub("world_model", "hyper_cez")


def test_is_efficient_zero_world_model_subclass(
    hypercez_world_model: HyperCEZWorldModel,
) -> None:
    assert isinstance(hypercez_world_model, EfficientZeroWorldModel)
    assert isinstance(hypercez_world_model, HyperCEZWorldModel)


def test_planner_accepts_hypercez_world_model(
    cartpole_training_env: TrainingEnv,
    hypercez_world_model: HyperCEZWorldModel,
) -> None:
    context = ComponentContext(
        backend=get_backend("jax"),
        config=hypercez_world_model.config,
        env=cartpole_training_env,
        world_model=hypercez_world_model,
    )
    planner = EfficientZeroPlanner(context)
    assert planner.world_model is hypercez_world_model
    assert planner.params is hypercez_world_model.params


def test_encode_and_initial_step(hypercez_world_model: HyperCEZWorldModel) -> None:
    obs = jnp.zeros((4,), dtype=jnp.float32)
    latent = hypercez_world_model.encode(obs)
    assert latent.state.shape == (hypercez_world_model.config.hidden_shape,)

    latent, value, policy = hypercez_world_model.initial_step(obs, training=True)
    assert value.shape == (
        hypercez_world_model.config.v_num,
        hypercez_world_model.config.support_bins,
    )
    assert policy.shape == (hypercez_world_model.env.action_dim * 2,)


def test_set_task_id_changes_materialized_params(
    hypercez_world_model: HyperCEZWorldModel,
) -> None:
    params0 = hypercez_world_model.set_task_id(0)
    params1 = hypercez_world_model.set_task_id(1)
    # Generated leaves should differ across task embeddings.
    gen0, _ = partition_params(params0["dynamics_model"])
    gen1, _ = partition_params(params1["dynamics_model"])
    assert not all(
        jnp.allclose(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(gen0),
            jax.tree_util.tree_leaves(gen1),
            strict=True,
        )
    )


def test_refresh_uses_live_shared_leaves(
    hypercez_world_model: HyperCEZWorldModel,
) -> None:
    hypercez_world_model.live_ez["representation_model"]["running_mean"] = (
        hypercez_world_model.live_ez["representation_model"]["running_mean"] + 1.0
    )
    hypercez_world_model.refresh_params()
    assert jnp.allclose(
        hypercez_world_model.params["representation_model"]["running_mean"],
        hypercez_world_model.live_ez["representation_model"]["running_mean"],
    )


def test_rejects_non_hypercez_config(cartpole_training_env: TrainingEnv) -> None:
    context = ComponentContext(
        backend=get_backend("jax"),
        config=BaseAgentConfig(),
        env=cartpole_training_env,
    )
    with pytest.raises(TypeError, match="HyperCEZConfig"):
        build_hyper_cez_world_model(context)


def test_chunked_world_model_materializes(cartpole_training_env: TrainingEnv) -> None:
    from algorl.backends.jax.nn.hypercez.hyper_model import ChunkedHyperNetwork

    config = HyperCEZConfig.for_dmc_state(
        num_tasks=3,
        emb_size=8,
        hnet_arch=(16, 16),
        hnet_type="chunked",
        chunk_dim=512,
        cemb_size=8,
        head_init_std=1e-3,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    world_model = build_hyper_cez_world_model(context)
    assert all(
        isinstance(module, ChunkedHyperNetwork)
        for module in world_model.hnet_modules.values()
    )
    assert "chunk_embeddings" in world_model.hnet_params["dynamics_model"]
    params0 = world_model.set_task_id(0)
    params1 = world_model.set_task_id(1)
    gen0, _ = partition_params(params0["dynamics_model"])
    gen1, _ = partition_params(params1["dynamics_model"])
    assert not all(
        jnp.allclose(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(gen0),
            jax.tree_util.tree_leaves(gen1),
            strict=True,
        )
    )
