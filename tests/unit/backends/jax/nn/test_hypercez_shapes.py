"""Tests for HyperCEZ generated/shared param partitioning."""

from __future__ import annotations

import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest
from flax.traverse_util import flatten_dict

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn import (
    build_efficient_zero_model_from_env,
    init_efficient_zero_params_from_env,
)
from algorl.backends.jax.nn.hypercez.shapes import (
    is_shared_leaf,
    merge_params,
    partition_params,
    path_strings,
    target_shapes,
)
from algorl.envs.training_env import TrainingEnv

HNET_COMPONENTS = (
    "representation_model",
    "dynamics_model",
    "reward_prediction_model",
    "value_policy_model",
)


@pytest.fixture
def cartpole_ez_params() -> dict:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, env)
    return init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), env)


def test_is_shared_leaf_matches_layernorm_and_obs_norm() -> None:
    assert is_shared_leaf(("LayerNorm_0", "scale"))
    assert is_shared_leaf(("ImproveResidualBlock_0", "LayerNorm_0", "bias"))
    assert is_shared_leaf(("running_mean",))
    assert is_shared_leaf(("running_var",))
    assert not is_shared_leaf(("Dense_0", "kernel"))
    assert not is_shared_leaf(("HyperMLP_0", "Dense_0", "bias"))
    assert not is_shared_leaf(("policy_head", "Dense_0", "kernel"))


def test_representation_partition_excludes_obs_norm_and_layernorm(
    cartpole_ez_params: dict,
) -> None:
    generated, shared = partition_params(cartpole_ez_params["representation_model"])
    generated_paths = path_strings(generated)
    shared_paths = path_strings(shared)

    assert "running_mean" in shared_paths
    assert "running_var" in shared_paths
    assert any("LayerNorm" in path for path in shared_paths)
    assert not any("LayerNorm" in path for path in generated_paths)
    assert not any(path.startswith("running_") for path in generated_paths)
    assert any("Dense" in path for path in generated_paths)


def test_merge_roundtrip_all_hnet_components(cartpole_ez_params: dict) -> None:
    for name in HNET_COMPONENTS:
        original = cartpole_ez_params[name]
        generated, shared = partition_params(original)
        merged = merge_params(generated, shared)
        assert path_strings(merged) == path_strings(original)
        for path, value in flatten_dict(original).items():
            assert jnp.allclose(flatten_dict(merged)[path], value)


def test_target_shapes_match_generated_leaves(cartpole_ez_params: dict) -> None:
    for name in HNET_COMPONENTS:
        component = cartpole_ez_params[name]
        generated, _ = partition_params(component)
        leaves, _ = jax.tree_util.tree_flatten(generated)
        shapes = target_shapes(component)
        assert shapes == tuple(tuple(leaf.shape) for leaf in leaves)
        assert len(shapes) > 0


def test_hypermlp_name_is_not_treated_as_shared(cartpole_ez_params: dict) -> None:
    """``HyperMLP`` is an ordinary MLP block name, not a hypernetwork."""
    generated, shared = partition_params(cartpole_ez_params["reward_prediction_model"])
    generated_paths = path_strings(generated)
    shared_paths = path_strings(shared)
    assert any(path.startswith("HyperMLP_0/") for path in generated_paths)
    assert not any("HyperMLP" in path for path in shared_paths)
