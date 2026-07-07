"""Continuous candidate-set MCTS planner tests."""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import jax
import jax.numpy as jnp
import mctx
import pytest
from mctx._src import search as mctx_search

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.planners.mcts.continuous import (
    ContinuousActionSelection,
    ContinuousSearchConfig,
    ContinuousSearchExtraData,
    ContinuousSearchResult,
    ContinuousSearchState,
    _initial_extra_data,
    _initialize_root_selection,
    _root_improved_policy,
    _to_recurrent_embedding,
    _update_min_max_stats_unbatched,
    build_continuous_root,
    build_continuous_root_from_model,
    initial_step_fn_from_world_model,
    make_continuous_recurrent_fn,
    make_jitted_continuous_search,
    make_model_continuous_recurrent_fn,
    recurrent_step_fn_from_world_model,
    run_continuous_search,
    sample_actions,
)
from algorl.backends.jax.world_models.efficientzero import (
    EfficientZeroLatentState,
    build_efficient_zero_world_model,
)
from algorl.core.component_context import ComponentContext
from algorl.core.factory import get_backend
from algorl.envs.training_env import TrainingEnv


def _stub_search_config(**overrides: object) -> ContinuousSearchConfig:
    defaults: dict[str, object] = {
        "num_simulations": 8,
        "num_sampled_actions": 4,
        "leaf_action_num": 2,
        "policy_action_num": 2,
        "random_action_num": 2,
        "num_top_actions": 4,
        "use_gumbel_noise": False,
    }
    defaults.update(overrides)
    return ContinuousSearchConfig(**defaults)


def _build_stub_root(
    config: ContinuousSearchConfig,
    gumbel_scores: list[float],
    *,
    batch_size: int = 1,
) -> tuple[mctx.RootFnOutput, jnp.ndarray, ContinuousSearchExtraData]:
    num_actions = config.num_sampled_actions
    candidates = jnp.arange(num_actions, dtype=jnp.float32)[None, :, None] * 0.1
    if batch_size > 1:
        candidates = jnp.tile(candidates, (batch_size, 1, 1))
    embedding = ContinuousSearchState(
        latent_state=jnp.zeros((batch_size, 1), dtype=jnp.float32),
        candidates=candidates,
        depth=jnp.zeros((batch_size,), dtype=jnp.int32),
    )
    gumbel = jnp.asarray(gumbel_scores, dtype=jnp.float32)[None, :]
    if batch_size > 1:
        gumbel = jnp.tile(gumbel, (batch_size, 1))
    extra_data = _initial_extra_data(gumbel, config)
    root = mctx.RootFnOutput(
        prior_logits=jnp.zeros((batch_size, num_actions), dtype=jnp.float32),
        value=jnp.full((batch_size,), 0.5, dtype=jnp.float32),
        embedding=_to_recurrent_embedding(embedding),
    )
    return root, candidates, extra_data


def _tree_batch_slice(tree: mctx.Tree, batch_index: int = 0) -> mctx.Tree:
    return jax.tree.map(lambda x: x[batch_index], tree)


def _stub_recurrent_step(
    latent_state: EfficientZeroLatentState,
    action: jnp.ndarray,
) -> tuple[EfficientZeroLatentState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del latent_state
    action_index = jnp.round(action[0] * 10.0).astype(jnp.int32)
    reward = action_index.astype(jnp.float32) * 0.1
    value = 0.2 + action_index.astype(jnp.float32) * 0.3
    next_latent = EfficientZeroLatentState(
        state=jnp.asarray([action_index], dtype=jnp.float32),
    )
    policy = jnp.array([0.0, 1e-6], dtype=jnp.float32)
    return next_latent, reward, value, policy


def _run_stub_search(
    config: ContinuousSearchConfig,
    gumbel_scores: list[float],
    rng_key: jnp.ndarray,
) -> ContinuousSearchResult:
    root, root_candidates, extra_data = _build_stub_root(config, gumbel_scores)
    recurrent_fn = make_continuous_recurrent_fn(_stub_recurrent_step, config=config)
    return run_continuous_search(
        params={},
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        config=config,
        extra_data=extra_data,
        root_candidates=root_candidates,
    )


@pytest.fixture
def cartpole_training_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


@pytest.fixture
def world_model(cartpole_training_env: TrainingEnv):
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    return build_efficient_zero_world_model(context)


def test_sample_actions_matches_efficientzero_v2_shapes() -> None:
    config = ContinuousSearchConfig(
        num_sampled_actions=8,
        policy_action_num=4,
        random_action_num=4,
    )
    policy = jnp.zeros((2, 4), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    actions, log_probs = sample_actions(policy, rng, config=config)
    assert actions.shape == (2, 8, 2)
    assert log_probs.shape == (2, 8)


def test_build_continuous_root(world_model) -> None:
    config = ContinuousSearchConfig(
        num_simulations=4,
        num_sampled_actions=8,
        policy_action_num=4,
        random_action_num=4,
    )
    rng = jax.random.PRNGKey(0)
    root, candidates, extra = build_continuous_root(
        jnp.zeros((4,), dtype=jnp.float32),
        initial_step_fn=initial_step_fn_from_world_model(world_model),
        config=config,
        rng=rng,
    )
    assert root.value.shape == (1,)
    assert root.prior_logits.shape == (1, 8)
    assert candidates.shape == (1, 8, world_model.env.action_dim)
    assert extra.gumbel.shape == (1, 8)


def test_run_continuous_search_with_world_model(world_model) -> None:
    config = ContinuousSearchConfig(
        num_simulations=4,
        num_sampled_actions=8,
        num_top_actions=8,
        policy_action_num=4,
        random_action_num=4,
    )
    rng = jax.random.PRNGKey(1)
    obs = jnp.zeros((4,), dtype=jnp.float32)
    root, root_candidates, extra = build_continuous_root(
        obs,
        initial_step_fn=initial_step_fn_from_world_model(world_model),
        config=config,
        rng=rng,
    )
    recurrent = make_model_continuous_recurrent_fn(world_model.model, config=config)
    rng, search_key = jax.random.split(rng)
    result = run_continuous_search(
        params=world_model.params,
        rng_key=search_key,
        root=root,
        recurrent_fn=recurrent,
        config=config,
        extra_data=extra,
        root_candidates=root_candidates,
    )
    assert result.action_indices.shape == (1,)
    assert result.action_weights.shape == (1, 8)
    assert jnp.isclose(jnp.sum(result.action_weights[0]), 1.0)
    assert result.actions.shape[-1] == world_model.env.action_dim
    assert jnp.isfinite(result.root_values).all()


def test_initialize_root_selection_orders_by_gumbel() -> None:
    config = _stub_search_config(num_simulations=4)
    root, _, extra_data = _build_stub_root(config, [3.0, 2.0, 1.0, 0.0])
    invalid_actions = jnp.zeros((1, config.num_sampled_actions), dtype=jnp.float32)
    tree = mctx_search.instantiate_tree_from_root(
        root,
        config.num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=extra_data,
    )
    initialized = _initialize_root_selection(tree, extra_data, config)
    assert int(initialized.num_selected[0]) == 4
    assert jnp.all(initialized.selected_children[0, :4] == jnp.array([0, 1, 2, 3], dtype=jnp.int32))


def test_continuous_root_action_selection_picks_least_visited() -> None:
    config = _stub_search_config(num_simulations=4)
    root, _, extra_data = _build_stub_root(config, [0.0, 0.0, 0.0, 0.0])
    invalid_actions = jnp.zeros((1, config.num_sampled_actions), dtype=jnp.float32)
    tree = mctx_search.instantiate_tree_from_root(
        root,
        config.num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=extra_data,
    )
    tree = tree.replace(extra_data=_initialize_root_selection(tree, extra_data, config))
    tree = tree.replace(
        children_visits=tree.children_visits.at[0, mctx.Tree.ROOT_INDEX].set(
            jnp.array([0, 5, 1, 1], dtype=jnp.int32),
        ),
    )
    selector = ContinuousActionSelection(config)
    action = selector.root(
        jnp.zeros((), dtype=jnp.uint32),
        _tree_batch_slice(tree),
        jnp.array(mctx.Tree.ROOT_INDEX, dtype=jnp.int32),
    )
    assert int(action) == 0


def test_continuous_interior_action_selection_picks_best_score() -> None:
    config = _stub_search_config(num_simulations=4)
    root, _, extra_data = _build_stub_root(config, [0.0, 0.0, 0.0, 0.0])
    invalid_actions = jnp.zeros((1, config.num_sampled_actions), dtype=jnp.float32)
    tree = mctx_search.instantiate_tree_from_root(
        root,
        config.num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=extra_data,
    )
    extra_data = _initialize_root_selection(tree, extra_data, config)
    extra_data = replace(
        extra_data,
        min_max_maximum=jnp.array([1.0], dtype=jnp.float32),
        min_max_minimum=jnp.array([0.0], dtype=jnp.float32),
    )
    tree = tree.replace(
        extra_data=extra_data,
        children_prior_logits=tree.children_prior_logits.at[0, 1, :2].set(
            jnp.array([-5.0, 5.0], dtype=jnp.float32),
        ),
        children_values=tree.children_values.at[0, 1, :2].set(
            jnp.array([0.1, 0.9], dtype=jnp.float32),
        ),
        children_rewards=tree.children_rewards.at[0, 1, :2].set(
            jnp.zeros(2, dtype=jnp.float32),
        ),
        children_visits=tree.children_visits.at[0, 1, :2].set(
            jnp.array([3, 0], dtype=jnp.int32),
        ),
        node_visits=tree.node_visits.at[0, 1].set(5),
        children_index=tree.children_index.at[0, 1, 0].set(2).at[0, 1, 1].set(mctx.Tree.UNVISITED),
        node_values=tree.node_values.at[0, 2].set(0.95),
    )
    selector = ContinuousActionSelection(config)
    action = selector.interior(
        jnp.zeros((), dtype=jnp.uint32),
        _tree_batch_slice(tree),
        jnp.array(1, dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
    )
    assert int(action) == 1


def test_run_continuous_search_stub_sequential_halving_and_gumbel_winner() -> None:
    config = _stub_search_config(num_simulations=8)
    result = _run_stub_search(config, [10.0, 1.0, 1.0, 1.0], jax.random.PRNGKey(42))
    extra = result.search_tree.extra_data
    assert int(result.action_indices[0]) == 0
    assert int(extra.num_selected[0]) == 1
    assert int(extra.selected_children[0, 0]) == 0
    assert int(extra.current_phase[0]) >= 1


def test_run_continuous_search_stub_allocates_root_visit_budget() -> None:
    config = _stub_search_config(num_simulations=8)
    result = _run_stub_search(config, [0.0, 0.0, 0.0, 0.0], jax.random.PRNGKey(7))
    root_child_visits = result.search_tree.children_visits[0, mctx.Tree.ROOT_INDEX]
    assert int(jnp.sum(root_child_visits)) == config.num_simulations
    assert int(jnp.max(result.search_tree.node_visits[0])) > 1


def test_run_continuous_search_stub_action_weights_match_improved_policy() -> None:
    config = _stub_search_config(num_simulations=8)
    result = _run_stub_search(config, [2.0, 1.0, 0.5, 0.0], jax.random.PRNGKey(11))
    expected = _root_improved_policy(result.search_tree, result.search_tree.extra_data, config)
    assert jnp.allclose(result.action_weights, expected, atol=1e-6)


def test_run_continuous_search_is_deterministic_with_fixed_rng() -> None:
    config = _stub_search_config(num_simulations=8)
    gumbel_scores = [4.0, 3.0, 2.0, 1.0]
    rng_key = jax.random.PRNGKey(99)
    first = _run_stub_search(config, gumbel_scores, rng_key)
    second = _run_stub_search(config, gumbel_scores, rng_key)
    assert int(first.action_indices[0]) == int(second.action_indices[0])
    assert jnp.allclose(first.action_weights, second.action_weights)
    assert jnp.allclose(
        first.search_tree.children_visits,
        second.search_tree.children_visits,
    )


def test_update_min_max_stats_tracks_backup_path() -> None:
    config = _stub_search_config(num_simulations=4)
    root, _, extra_data = _build_stub_root(config, [0.0, 0.0, 0.0, 0.0])
    invalid_actions = jnp.zeros((1, config.num_sampled_actions), dtype=jnp.float32)
    tree = mctx_search.instantiate_tree_from_root(
        root,
        config.num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=extra_data,
    )
    tree = tree.replace(
        parents=tree.parents.at[0, 1].set(mctx.Tree.ROOT_INDEX),
        action_from_parent=tree.action_from_parent.at[0, 1].set(0),
        children_rewards=tree.children_rewards.at[0, 0, 0].set(0.1),
        children_discounts=tree.children_discounts.at[0, 0, 0].set(0.997),
        node_values=tree.node_values.at[0, 1].set(0.8),
    )
    updated = _update_min_max_stats_unbatched(
        jax.tree.map(lambda x: x[0], tree),
        replace(
            extra_data,
            gumbel=extra_data.gumbel[0],
            min_max_maximum=extra_data.min_max_maximum[0],
            min_max_minimum=extra_data.min_max_minimum[0],
            selected_children=extra_data.selected_children[0],
            num_selected=extra_data.num_selected[0],
            current_num_top_actions=extra_data.current_num_top_actions[0],
            current_phase=extra_data.current_phase[0],
            visit_num_for_next_phase=extra_data.visit_num_for_next_phase[0],
            used_visit_num=extra_data.used_visit_num[0],
        ),
        jnp.array(1, dtype=jnp.int32),
    )
    expected = 0.1 + 0.997 * 0.8
    assert float(updated.min_max_maximum) == pytest.approx(expected)
    assert float(updated.min_max_minimum) == pytest.approx(expected)


def test_sample_actions_is_jittable() -> None:
    config = ContinuousSearchConfig(
        num_sampled_actions=8,
        policy_action_num=4,
        random_action_num=4,
    )
    policy = jnp.zeros((1, 4), dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    jit_sample = jax.jit(lambda p, k: sample_actions(p, k, config=config))
    actions, log_probs = jit_sample(policy, rng)
    assert actions.shape == (1, 8, 2)
    assert log_probs.shape == (1, 8)


def test_build_continuous_root_from_model_is_jittable(world_model) -> None:
    config = ContinuousSearchConfig(
        num_simulations=4,
        num_sampled_actions=8,
        policy_action_num=4,
        random_action_num=4,
    )
    obs = jnp.zeros((4,), dtype=jnp.float32)
    rng = jax.random.PRNGKey(3)
    jit_build = jax.jit(
        lambda p, o, k: build_continuous_root_from_model(
            world_model.model,
            p,
            o,
            config=config,
            rng=k,
        ),
    )
    root, candidates, extra = jit_build(world_model.params, obs, rng)
    assert root.value.shape == (1,)
    assert candidates.shape == (1, 8, world_model.env.action_dim)
    assert extra.gumbel.shape == (1, 8)


def test_jitted_continuous_search_matches_eager(world_model) -> None:
    config = _stub_search_config(
        num_simulations=4,
        num_sampled_actions=8,
        num_top_actions=8,
        policy_action_num=4,
        random_action_num=4,
    )
    recurrent_fn = make_model_continuous_recurrent_fn(world_model.model, config=config)
    jitted_search = make_jitted_continuous_search(config, recurrent_fn)

    rng = jax.random.PRNGKey(5)
    obs = jnp.zeros((4,), dtype=jnp.float32)
    eager_root, root_candidates, extra = build_continuous_root(
        obs,
        initial_step_fn=initial_step_fn_from_world_model(world_model),
        config=config,
        rng=rng,
    )
    rng, eager_key = jax.random.split(rng)
    eager_result = run_continuous_search(
        params=world_model.params,
        rng_key=eager_key,
        root=eager_root,
        recurrent_fn=recurrent_fn,
        config=config,
        extra_data=extra,
        root_candidates=root_candidates,
    )

    rng, jit_key = jax.random.split(rng)
    jit_result = jitted_search(
        world_model.params,
        jit_key,
        eager_root,
        extra,
        root_candidates,
    )
    assert int(jit_result.action_indices[0]) == int(eager_result.action_indices[0])
    assert jnp.allclose(jit_result.action_weights, eager_result.action_weights, atol=1e-5)
    assert jnp.allclose(jit_result.root_values, eager_result.root_values, atol=1e-5)


def test_run_continuous_search_batch_independent_envs() -> None:
    config = _stub_search_config(num_simulations=8)
    batch_size = 2
    root, root_candidates, extra_data = _build_stub_root(
        config,
        [10.0, 1.0, 1.0, 1.0],
        batch_size=batch_size,
    )
    recurrent_fn = make_continuous_recurrent_fn(_stub_recurrent_step, config=config)
    rng_key = jax.random.PRNGKey(21)
    result = run_continuous_search(
        params={},
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        config=config,
        extra_data=extra_data,
        root_candidates=root_candidates,
    )
    assert result.action_indices.shape == (batch_size,)
    assert result.actions.shape == (batch_size, 1)
    assert result.action_weights.shape == (batch_size, config.num_sampled_actions)
    assert result.root_values.shape == (batch_size,)
    assert result.root_candidates.shape == (batch_size, config.num_sampled_actions, 1)
    assert int(result.action_indices[0]) == 0
    assert int(result.search_tree.children_visits.shape[0]) == batch_size
    per_env_visits = jnp.sum(
        result.search_tree.children_visits[:, mctx.Tree.ROOT_INDEX],
        axis=-1,
    )
    assert jnp.all(per_env_visits == config.num_simulations)
