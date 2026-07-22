"""EfficientZero learner tests."""

from __future__ import annotations

import copy
from unittest import mock

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.learners.efficientzero import (
    EfficientZeroLearner,
    build_efficient_zero_learner,
)
from algorl.backends.jax.planners.efficientzero import build_efficient_zero_planner
from algorl.backends.jax.world_models.efficientzero import build_efficient_zero_world_model
from algorl.buffers.efficientzero import (
    BEST_ACTION_INFO_KEY,
    POLICY_TARGET_INFO_KEY,
    ROOT_CANDIDATES_INFO_KEY,
    SEARCH_VALUE_INFO_KEY,
    EfficientZeroReplayBuffer,
)
from algorl.core.component_context import ComponentContext
from algorl.core.factory import get_backend
from algorl.core.types import Transition
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_training_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


@pytest.fixture
def learner_context(cartpole_training_env: TrainingEnv) -> ComponentContext:
    config = EfficientZeroConfig.for_dmc_state(
        require_implemented=False,
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        learning_starts=0,
        mcts_simulations=2,
        reanalyze_ratio=0.0,
        use_priority=False,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    context.world_model = build_efficient_zero_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    return context


def _fill_buffer(buffer: EfficientZeroReplayBuffer, *, policy_dim: int = 4) -> None:
    for index in range(8):
        policy = np.full((policy_dim,), 1.0 / policy_dim, dtype=np.float32)
        candidates = np.stack(
            [np.asarray([0.1], dtype=np.float32) for _ in range(policy_dim)],
            axis=0,
        )
        buffer.add(
            Transition(
                observation=np.full((4,), float(index), dtype=np.float32),
                action=np.asarray([0.1], dtype=np.float32),
                reward=float(index) * 0.1,
                next_observation=np.full((4,), float(index + 1), dtype=np.float32),
                done=index % 4 == 3,
                info={
                    POLICY_TARGET_INFO_KEY: policy,
                    SEARCH_VALUE_INFO_KEY: float(index) * 0.05,
                    ROOT_CANDIDATES_INFO_KEY: candidates,
                    BEST_ACTION_INFO_KEY: np.asarray([0.1], dtype=np.float32),
                },
            )
        )


def test_build_efficient_zero_learner(learner_context: ComponentContext) -> None:
    learner = build_efficient_zero_learner(learner_context)
    assert isinstance(learner, EfficientZeroLearner)


def test_efficient_zero_learner_train_step_updates_params(learner_context: ComponentContext) -> None:
    learner = build_efficient_zero_learner(learner_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=learner_context.config,
        unroll_steps=learner_context.config.unroll_steps,
        trajectory_size=learner_context.config.trajectory_size,
    )
    _fill_buffer(buffer)

    params_before = copy.deepcopy(jax.tree.map(np.asarray, learner.params))
    metrics = learner.train_step(buffer)

    assert np.isfinite(metrics["loss"])
    assert metrics["value_loss"] >= 0.0
    assert metrics["policy_loss"] >= 0.0
    assert learner.world_model.params is learner.params
    assert learner.planner.params is learner.params

    params_after = jax.tree.map(np.asarray, learner.params)
    changed = any(
        not np.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(params_before),
            jax.tree.leaves(params_after),
            strict=True,
        )
    )
    assert changed


def test_efficient_zero_learner_train_step_with_large_batch(learner_context: ComponentContext) -> None:
    config = learner_context.config.with_overrides(batch_size=8, unroll_steps=5, trajectory_size=12)
    context = ComponentContext(
        backend=learner_context.backend,
        config=config,
        env=learner_context.env,
    )
    context.world_model = build_efficient_zero_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    learner = build_efficient_zero_learner(context)
    buffer = EfficientZeroReplayBuffer(
        capacity=200,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    for index in range(80):
        policy = np.full((4,), 0.25, dtype=np.float32)
        candidates = np.zeros((4, 1), dtype=np.float32)
        buffer.add(
            Transition(
                observation=np.full((4,), float(index), dtype=np.float32),
                action=np.asarray([0.1], dtype=np.float32),
                reward=float(index) * 0.1,
                next_observation=np.full((4,), float(index + 1), dtype=np.float32),
                done=index % 6 == 5,
                info={
                    POLICY_TARGET_INFO_KEY: policy,
                    SEARCH_VALUE_INFO_KEY: float(index) * 0.05,
                    ROOT_CANDIDATES_INFO_KEY: candidates,
                    BEST_ACTION_INFO_KEY: np.asarray([0.1], dtype=np.float32),
                },
            )
        )
    metrics = learner.train_step(buffer)
    assert np.isfinite(metrics["loss"])


def test_efficient_zero_learner_train_burst_reanalyze_once(learner_context: ComponentContext) -> None:
    config = learner_context.config.with_overrides(
        batch_size=2,
        reanalyze_ratio=1.0,
        gradient_steps_per_rollout=3,
    )
    context = ComponentContext(
        backend=learner_context.backend,
        config=config,
        env=learner_context.env,
    )
    context.world_model = build_efficient_zero_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    learner = build_efficient_zero_learner(context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    _fill_buffer(buffer)

    with mock.patch(
        "algorl.backends.jax.learners.efficientzero.learner.reanalyze_fused_policy_batches",
        return_value=[
            (
                np.zeros((2, config.unroll_steps + 1, 4), dtype=np.float32),
                np.zeros((2, config.unroll_steps + 1), dtype=np.float32),
                np.zeros((2, config.unroll_steps + 1, 4, 1), dtype=np.float32),
                np.zeros((2, config.unroll_steps + 1, 1), dtype=np.float32),
            )
        ]
        * 3,
    ) as mock_reanalyze:
        metrics = learner.train_burst(buffer, 3)

    assert np.isfinite(metrics["loss"])
    assert mock_reanalyze.call_count == 1
    assert len(mock_reanalyze.call_args.args[1]) == 3


def test_efficient_zero_learner_train_burst_reanalyzes_per_sub_burst(
    learner_context: ComponentContext,
) -> None:
    """Chunked bursts re-sample and re-reanalyze each sub-burst with fresh params.

    Freezing sampling/reanalyze across the whole burst trains later steps
    toward stale targets and priorities, hurting sample efficiency.
    """
    config = learner_context.config.with_overrides(
        batch_size=2,
        reanalyze_ratio=1.0,
        gradient_steps_per_rollout=6,
        burst_compile_steps=2,
    )
    context = ComponentContext(
        backend=learner_context.backend,
        config=config,
        env=learner_context.env,
    )
    context.world_model = build_efficient_zero_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    learner = build_efficient_zero_learner(context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    _fill_buffer(buffer)

    with mock.patch(
        "algorl.backends.jax.learners.efficientzero.learner.reanalyze_fused_policy_batches",
        return_value=[
            (
                np.zeros((2, config.unroll_steps + 1, 4), dtype=np.float32),
                np.zeros((2, config.unroll_steps + 1), dtype=np.float32),
                np.zeros((2, config.unroll_steps + 1, 4, 1), dtype=np.float32),
                np.zeros((2, config.unroll_steps + 1, 1), dtype=np.float32),
            )
        ]
        * 2,
    ) as mock_reanalyze:
        metrics = learner.train_burst(buffer, 6)

    assert np.isfinite(metrics["loss"])
    # One fused reanalyze per 2-step sub-burst.
    assert mock_reanalyze.call_count == 3
    assert all(len(call.args[1]) == 2 for call in mock_reanalyze.call_args_list)
    assert learner._train_steps == 6


def test_efficient_zero_learner_skips_optimizer_on_non_finite_loss(
    learner_context: ComponentContext,
) -> None:
    learner = build_efficient_zero_learner(learner_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=learner_context.config,
        unroll_steps=learner_context.config.unroll_steps,
        trajectory_size=learner_context.config.trajectory_size,
    )
    _fill_buffer(buffer)
    params_before = copy.deepcopy(jax.tree.map(np.asarray, learner.params))

    with mock.patch(
        "algorl.backends.jax.learners.efficientzero.learner._loss_from_batch",
        return_value=(
            jnp.asarray(float("nan"), dtype=jnp.float32),
            {"priorities": jnp.full((2,), float("nan"), dtype=jnp.float32)},
        ),
    ):
        metrics = learner.train_step(buffer)

    params_after = jax.tree.map(np.asarray, learner.params)
    assert not np.isfinite(metrics["loss"])
    assert all(
        np.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(params_before),
            jax.tree.leaves(params_after),
            strict=True,
        )
    )


def test_sync_self_play_for_rollout_refreshes_planner_copy(learner_context: ComponentContext) -> None:
    context = learner_context
    context.world_model = build_efficient_zero_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    learner = build_efficient_zero_learner(context)
    planner = context.planner

    before_self_play = jax.tree.map(np.asarray, planner.self_play_params)
    learner.params = jax.tree.map(lambda leaf: leaf + 1.0, learner.params)

    learner.sync_self_play_for_rollout()

    after_self_play = jax.tree.map(np.asarray, planner.self_play_params)
    learner_weights = jax.tree.map(np.asarray, learner.params)
    assert planner.self_play_params is learner._self_play_params
    assert not np.allclose(
        jax.tree.leaves(before_self_play)[0],
        jax.tree.leaves(after_self_play)[0],
    )
    assert np.allclose(
        jax.tree.leaves(after_self_play)[0],
        jax.tree.leaves(learner_weights)[0],
    )


def test_batched_search_outputs_normalizes_mcts_shapes() -> None:
    from algorl.backends.jax.learners.efficientzero.reanalyze import _batched_search_outputs

    class _Result:
        action_weights = np.ones((4, 16), dtype=np.float32)
        root_values = np.arange(4, dtype=np.float32)
        root_candidates = np.zeros((4, 16), dtype=np.float32)  # 1-D actions
        actions = np.zeros((4,), dtype=np.float32)

    weights, values, candidates, best = _batched_search_outputs(_Result(), valid_count=3)
    assert weights.shape == (3, 16)
    assert values.shape == (3,)
    assert candidates.shape == (3, 16, 1)
    assert best.shape == (3, 1)
    assert values.tolist() == [0.0, 1.0, 2.0]
