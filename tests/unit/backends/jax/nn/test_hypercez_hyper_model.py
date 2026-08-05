"""Tests for HyperCEZ Flax hypernetwork forward and helpers."""

from __future__ import annotations

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn import (
    build_efficient_zero_model_from_env,
    init_efficient_zero_params_from_env,
)
from algorl.backends.jax.nn.hypercez.hyper_model import (
    ChunkedHyperNetwork,
    apply_hypernetwork,
    build_hypernetwork_for_component,
    component_outputs_to_tree,
    init_hypernetwork_params,
)
from algorl.backends.jax.nn.hypercez.shapes import partition_params, target_shapes
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_ez_params() -> dict:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, env)
    return init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), env)


def test_hypernetwork_output_shapes_match_targets(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["dynamics_model"]
    shapes = target_shapes(component)
    module = build_hypernetwork_for_component(component, num_tasks=3, emb_size=8)
    params = init_hypernetwork_params(module, jax.random.PRNGKey(1))
    outputs = apply_hypernetwork(module, params, task_id=0)

    assert len(outputs) == len(shapes)
    for output, shape in zip(outputs, shapes, strict=True):
        assert tuple(output.shape) == shape


def test_zero_init_heads_yield_zero_deltas(cartpole_ez_params: dict) -> None:
    """Identity residual: ΔW(0)=0 so W(0)=W0 even with large α."""
    component = cartpole_ez_params["representation_model"]
    for hnet_type in ("unchunked", "chunked"):
        module = build_hypernetwork_for_component(
            component, num_tasks=4, emb_size=8, hnet_type=hnet_type,  # type: ignore[arg-type]
            chunk_dim=64, cemb_size=8,
        )
        params = init_hypernetwork_params(module, jax.random.PRNGKey(2))
        outputs = apply_hypernetwork(module, params, task_id=0)
        for leaf in outputs:
            assert float(jnp.max(jnp.abs(leaf))) == 0.0


def test_different_task_embeddings_change_outputs_after_head_noise(
    cartpole_ez_params: dict,
) -> None:
    component = cartpole_ez_params["representation_model"]
    module = build_hypernetwork_for_component(component, num_tasks=4, emb_size=8)
    params = init_hypernetwork_params(module, jax.random.PRNGKey(2))

    def _nudge_heads(path: tuple, leaf: jnp.ndarray) -> jnp.ndarray:
        keys = [str(getattr(p, "key", p)) for p in path]
        if any(k.startswith("head_") or k == "chunk_head" for k in keys):
            return leaf + jax.random.normal(jax.random.PRNGKey(leaf.size), leaf.shape) * 0.05
        return leaf

    params = jax.tree_util.tree_map_with_path(_nudge_heads, params)
    out0 = apply_hypernetwork(module, params, task_id=0)
    out1 = apply_hypernetwork(module, params, task_id=1)
    assert not all(
        jnp.allclose(a, b) for a, b in zip(out0, out1, strict=True)
    )


def test_dtheta_shifts_hypernetwork_outputs(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["reward_prediction_model"]
    module = build_hypernetwork_for_component(component, num_tasks=2, emb_size=8)
    params = init_hypernetwork_params(module, jax.random.PRNGKey(3))
    baseline = apply_hypernetwork(module, params, task_id=0)

    dtheta = jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) * 0.01, params)
    shifted = apply_hypernetwork(module, params, task_id=0, dtheta=dtheta)
    assert not all(
        jnp.allclose(a, b) for a, b in zip(baseline, shifted, strict=True)
    )


def test_outputs_map_onto_generated_partition(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["value_policy_model"]
    module = build_hypernetwork_for_component(component, num_tasks=2, emb_size=8)
    params = init_hypernetwork_params(module, jax.random.PRNGKey(4))
    outputs = apply_hypernetwork(module, params, task_id=0)

    delta_tree = component_outputs_to_tree(outputs, component)
    generated, _ = partition_params(component)
    assert jax.tree_util.tree_structure(delta_tree) == jax.tree_util.tree_structure(
        generated
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(delta_tree),
        jax.tree_util.tree_leaves(generated),
        strict=True,
    ):
        assert left.shape == right.shape


def test_chunked_hypernetwork_output_shapes_match_targets(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["dynamics_model"]
    shapes = target_shapes(component)
    module = build_hypernetwork_for_component(
        component,
        num_tasks=3,
        emb_size=8,
        hnet_type="chunked",
        chunk_dim=256,
        cemb_size=8,
        hidden_dims=(16, 16),
    )
    assert isinstance(module, ChunkedHyperNetwork)
    params = init_hypernetwork_params(module, jax.random.PRNGKey(5))
    assert "chunk_embeddings" in params
    assert "task_embeddings" in params
    outputs = apply_hypernetwork(module, params, task_id=0)
    assert len(outputs) == len(shapes)
    for output, shape in zip(outputs, shapes, strict=True):
        assert tuple(output.shape) == shape


def test_chunked_hypernetwork_has_fewer_params_than_unchunked(
    cartpole_ez_params: dict,
) -> None:
    component = cartpole_ez_params["dynamics_model"]

    def count(tree: object) -> int:
        return sum(int(np.prod(np.asarray(leaf).shape)) for leaf in jax.tree_util.tree_leaves(tree))

    unchunked = build_hypernetwork_for_component(
        component, num_tasks=3, emb_size=8, hidden_dims=(64, 64)
    )
    chunked = build_hypernetwork_for_component(
        component,
        num_tasks=3,
        emb_size=8,
        hidden_dims=(20, 20),
        hnet_type="chunked",
        chunk_dim=512,
        cemb_size=16,
    )
    unchunked_n = count(init_hypernetwork_params(unchunked, jax.random.PRNGKey(6)))
    chunked_n = count(init_hypernetwork_params(chunked, jax.random.PRNGKey(7)))
    assert chunked_n < unchunked_n


def test_chunked_dtheta_shifts_outputs(cartpole_ez_params: dict) -> None:
    component = cartpole_ez_params["reward_prediction_model"]
    module = build_hypernetwork_for_component(
        component,
        num_tasks=2,
        emb_size=8,
        hnet_type="chunked",
        chunk_dim=128,
        cemb_size=8,
        hidden_dims=(16, 16),
    )
    params = init_hypernetwork_params(module, jax.random.PRNGKey(8))
    baseline = apply_hypernetwork(module, params, task_id=0)
    dtheta = jax.tree_util.tree_map(lambda leaf: jnp.ones_like(leaf) * 0.01, params)
    shifted = apply_hypernetwork(module, params, task_id=0, dtheta=dtheta)
    assert not all(jnp.allclose(a, b) for a, b in zip(baseline, shifted, strict=True))


def test_invalid_hnet_type_raises(cartpole_ez_params: dict) -> None:
    with pytest.raises(ValueError, match="hnet_type"):
        build_hypernetwork_for_component(
            cartpole_ez_params["dynamics_model"],
            hnet_type="bogus",  # type: ignore[arg-type]
        )
