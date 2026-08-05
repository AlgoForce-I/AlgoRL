"""Tests for HyperCEZ delta materialization."""

from __future__ import annotations

import copy

import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn import (
    build_efficient_zero_model_from_env,
    init_efficient_zero_params_from_env,
)
from algorl.backends.jax.nn.hypercez.hyper_model import (
    apply_hypernetwork,
    build_hypernetwork_for_component,
    component_outputs_to_tree,
    init_hypernetwork_params,
)
from algorl.backends.jax.nn.hypercez.materialize import (
    materialize_delta_component,
    materialize_ez_params,
)
from algorl.backends.jax.nn.hypercez.shapes import partition_params, path_strings
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_ez_params() -> dict:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, env)
    return init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), env)


def test_zero_delta_keeps_frozen_generated_leaves(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["dynamics_model"]
    frozen_generated, shared = partition_params(component)
    zero_delta = jax.tree_util.tree_map(jnp.zeros_like, frozen_generated)

    materialized = materialize_delta_component(
        delta_tree=zero_delta,
        frozen_base=component,
        shared_live=shared,
        alpha=0.5,
        alpha_max=0.2,
    )
    mat_generated, mat_shared = partition_params(materialized)
    for left, right in zip(
        jax.tree_util.tree_leaves(mat_generated),
        jax.tree_util.tree_leaves(frozen_generated),
        strict=True,
    ):
        assert jnp.allclose(left, right)
    for left, right in zip(
        jax.tree_util.tree_leaves(mat_shared),
        jax.tree_util.tree_leaves(shared),
        strict=True,
    ):
        assert jnp.allclose(left, right)


def test_scaled_delta_applies_alpha_max_tanh(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["representation_model"]
    frozen_generated, shared = partition_params(component)
    ones_delta = jax.tree_util.tree_map(jnp.ones_like, frozen_generated)
    alpha = jnp.asarray(2.0)
    alpha_max = 0.2
    expected_scale = float(alpha_max * jnp.tanh(alpha))

    materialized = materialize_delta_component(
        delta_tree=ones_delta,
        frozen_base=component,
        shared_live=shared,
        alpha=alpha,
        alpha_max=alpha_max,
    )
    mat_generated, _ = partition_params(materialized)
    for mat_leaf, base_leaf in zip(
        jax.tree_util.tree_leaves(mat_generated),
        jax.tree_util.tree_leaves(frozen_generated),
        strict=True,
    ):
        assert jnp.allclose(mat_leaf, base_leaf + expected_scale)


def test_materialize_uses_live_shared_not_frozen(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["representation_model"]
    frozen = copy.deepcopy(component)
    live = copy.deepcopy(component)
    live["running_mean"] = live["running_mean"] + 1.0
    live["LayerNorm_0"]["scale"] = live["LayerNorm_0"]["scale"] + 0.5

    frozen_generated, _ = partition_params(frozen)
    zero_delta = jax.tree_util.tree_map(jnp.zeros_like, frozen_generated)
    _, live_shared = partition_params(live)

    materialized = materialize_delta_component(
        delta_tree=zero_delta,
        frozen_base=frozen,
        shared_live=live_shared,
        alpha=0.0,
    )
    assert jnp.allclose(materialized["running_mean"], live["running_mean"])
    assert jnp.allclose(
        materialized["LayerNorm_0"]["scale"],
        live["LayerNorm_0"]["scale"],
    )


def test_materialize_ez_params_leaves_projections_untouched(
    cartpole_ez_params: dict,
) -> None:
    frozen = copy.deepcopy(cartpole_ez_params)
    live = copy.deepcopy(cartpole_ez_params)
    live["projection_model"]["Dense_0"]["bias"] = (
        live["projection_model"]["Dense_0"]["bias"] + 3.0
    )

    hnet_components = ("dynamics_model",)
    component = frozen["dynamics_model"]
    module = build_hypernetwork_for_component(component, num_tasks=2, emb_size=8)
    hnet_params = init_hypernetwork_params(module, jax.random.PRNGKey(5))
    outputs = apply_hypernetwork(module, hnet_params, task_id=0)
    delta_tree = component_outputs_to_tree(outputs, component)

    materialized = materialize_ez_params(
        component_deltas={"dynamics_model": delta_tree},
        frozen_ez=frozen,
        live_ez=live,
        alphas={"dynamics_model": 0.1},
        hnet_components=hnet_components,
        alpha_max=0.2,
    )

    assert path_strings(materialized["projection_model"]) == path_strings(
        live["projection_model"]
    )
    assert jnp.allclose(
        materialized["projection_model"]["Dense_0"]["bias"],
        live["projection_model"]["Dense_0"]["bias"],
    )
    assert "dynamics_model" in materialized
    assert path_strings(materialized["dynamics_model"]) == path_strings(
        live["dynamics_model"]
    )
