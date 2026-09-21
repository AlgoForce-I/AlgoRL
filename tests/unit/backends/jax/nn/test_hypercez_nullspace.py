"""Tests for null-space projected hypernet updates (HyperCEZ ``cl_strategy='nullspace'``)."""

from __future__ import annotations

import math

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
    apply_hypernetwork,
    build_hypernetwork_for_component,
    hypernetwork_layer_inputs,
    init_hypernetwork_params,
)
from algorl.backends.jax.nn.hypercez.nullspace import (
    build_nullspace_bases,
    free_direction_fraction,
    mask_alpha_updates,
    project_hnet_updates,
)
from algorl.envs.training_env import TrainingEnv

COMPONENT = "dynamics_model"


@pytest.fixture
def tiny_hnet() -> tuple:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, env)
    ez_params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), env)
    module = build_hypernetwork_for_component(
        ez_params[COMPONENT],
        hidden_dims=(16, 16),
        emb_size=4,
        num_tasks=6,
        head_init_std=0.1,
    )
    params = init_hypernetwork_params(module, jax.random.PRNGKey(1))
    return module, params


def _flat_outputs(module, params, task_id: int) -> np.ndarray:
    return np.concatenate(
        [np.ravel(np.asarray(leaf)) for leaf in apply_hypernetwork(module, params, task_id)]
    )


def _random_like(tree, seed: int):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [jax.random.normal(key, leaf.shape) * 0.3 for key, leaf in zip(keys, leaves, strict=True)],
    )


def test_layer_inputs_reproduce_head_outputs(tiny_hnet) -> None:
    module, params = tiny_hnet
    inputs = hypernetwork_layer_inputs(module, params, [0, 2])
    assert set(inputs) == {"hidden_0", "hidden_1", "trunk_norm", "heads"}
    assert inputs["hidden_0"].shape == (2, 4)
    np.testing.assert_allclose(inputs["hidden_0"], params["task_embeddings"][jnp.asarray([0, 2])])
    # Normalized trunk input, after scale/bias, is exactly the head input.
    rebuilt = inputs["trunk_norm"] * params["trunk_norm"]["scale"] + params["trunk_norm"]["bias"]
    np.testing.assert_allclose(rebuilt, inputs["heads"], atol=1e-5)
    head = params["head_0"]
    scale = 1.0 / math.sqrt(16)
    expected = (inputs["heads"][1] @ head["kernel"] + head["bias"]) * scale
    np.testing.assert_allclose(
        np.ravel(apply_hypernetwork(module, params, 2)[0]), expected, atol=1e-5
    )


def test_projected_update_keeps_protected_outputs_and_moves_current(tiny_hnet) -> None:
    module, params = tiny_hnet
    protected = [0, 1, 2]
    current = 3
    before = {task: _flat_outputs(module, params, task) for task in (*protected, current)}
    bases = build_nullspace_bases(
        {COMPONENT: params}, {COMPONENT: module}, (COMPONENT,), protected
    )
    update = _random_like(params, seed=7)
    projected = project_hnet_updates(
        {COMPONENT: update}, bases, task_id=current, hnet_components=(COMPONENT,)
    )[COMPONENT]
    new_params = jax.tree.map(lambda p, u: p + u, params, projected)

    for task in protected:
        drift = np.linalg.norm(_flat_outputs(module, new_params, task) - before[task])
        assert drift <= 1e-4 * max(1.0, np.linalg.norm(before[task])), task
    moved = np.linalg.norm(_flat_outputs(module, new_params, current) - before[current])
    assert moved > 1e-2
    embeddings = np.asarray(projected["task_embeddings"])
    assert np.all(embeddings[np.arange(6) != current] == 0.0)
    np.testing.assert_allclose(embeddings[current], np.asarray(update["task_embeddings"])[current])


def test_protection_survives_many_projected_steps(tiny_hnet) -> None:
    module, params = tiny_hnet
    protected = [0, 1]
    before = {task: _flat_outputs(module, params, task) for task in protected}
    bases = build_nullspace_bases(
        {COMPONENT: params}, {COMPONENT: module}, (COMPONENT,), protected
    )
    for step in range(50):
        update = _random_like(params, seed=100 + step)
        projected = project_hnet_updates(
            {COMPONENT: update}, bases, task_id=2, hnet_components=(COMPONENT,)
        )[COMPONENT]
        params = jax.tree.map(lambda p, u: p + 0.1 * u, params, projected)
    for task in protected:
        drift = np.linalg.norm(_flat_outputs(module, params, task) - before[task])
        assert drift <= 1e-3 * max(1.0, np.linalg.norm(before[task])), task


def test_no_protected_tasks_only_masks_embeddings(tiny_hnet) -> None:
    module, params = tiny_hnet
    assert build_nullspace_bases({COMPONENT: params}, {COMPONENT: module}, (COMPONENT,), []) is None
    update = _random_like(params, seed=3)
    projected = project_hnet_updates(
        {COMPONENT: update}, None, task_id=0, hnet_components=(COMPONENT,)
    )[COMPONENT]
    np.testing.assert_allclose(projected["hidden_1"]["kernel"], update["hidden_1"]["kernel"])
    assert np.all(np.asarray(projected["task_embeddings"])[1:] == 0.0)


def test_free_direction_fraction_bounds(tiny_hnet) -> None:
    module, params = tiny_hnet
    bases = build_nullspace_bases({COMPONENT: params}, {COMPONENT: module}, (COMPONENT,), [0, 1])
    fractions = free_direction_fraction(
        {COMPONENT: params}, {COMPONENT: module}, (COMPONENT,), bases, task_id=2
    )
    assert 0.0 < fractions[COMPONENT] <= 1.0
    protected = free_direction_fraction(
        {COMPONENT: params}, {COMPONENT: module}, (COMPONENT,), bases, task_id=1
    )
    assert protected[COMPONENT] < 1e-4


def test_mask_alpha_updates_keeps_only_current_task() -> None:
    updates = {
        task: {"a": jnp.asarray(1.0 + task), "b": jnp.asarray(-1.0)} for task in range(3)
    }
    masked = mask_alpha_updates(updates, task_id=1)
    assert float(masked[1]["a"]) == 2.0
    assert float(masked[0]["a"]) == 0.0 and float(masked[2]["b"]) == 0.0


def test_layer_inputs_reject_chunked_hypernet() -> None:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, env)
    ez_params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), env)
    module = build_hypernetwork_for_component(
        ez_params[COMPONENT], hnet_type="chunked", chunk_dim=64, cemb_size=8, num_tasks=3
    )
    params = init_hypernetwork_params(module, jax.random.PRNGKey(1))
    with pytest.raises(TypeError):
        hypernetwork_layer_inputs(module, params, [0])
