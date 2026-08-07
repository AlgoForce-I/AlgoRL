"""HyperCEZ fused burst and delayed hnet copy tests."""

from __future__ import annotations

import copy
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from algorl.agents.configs import HyperCEZConfig
from algorl.backends.jax.learners.hypercez import build_hyper_cez_learner
from algorl.backends.jax.planners.efficientzero import build_efficient_zero_planner
from algorl.backends.jax.world_models.hypercez import build_hyper_cez_world_model
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
def burst_context() -> ComponentContext:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = HyperCEZConfig.for_dmc_state(
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        learning_starts=0,
        mcts_simulations=2,
        reanalyze_ratio=0.0,
        use_priority=False,
        num_tasks=3,
        emb_size=8,
        hnet_arch=(32, 32),
        burst_compile_steps=2,
        gradient_steps_per_rollout=4,
        lr_warm_up=0.0,
        self_play_update_interval=1,
        reanalyze_update_interval=1,
        head_init_std=1e-3,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=env,
    )
    context.world_model = build_hyper_cez_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    return context


def _fill_buffer(buffer: EfficientZeroReplayBuffer, *, policy_dim: int = 4) -> None:
    for index in range(12):
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


def test_hypercez_train_burst_runs_compiled_scan(burst_context: ComponentContext) -> None:
    learner = build_hyper_cez_learner(burst_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=burst_context.config,
        unroll_steps=burst_context.config.unroll_steps,
        trajectory_size=burst_context.config.trajectory_size,
    )
    _fill_buffer(buffer)

    metrics = learner.train_burst(buffer, 4)
    assert np.isfinite(metrics["loss"])
    assert learner._train_steps == 4


def test_delayed_reanalyze_hnets_lag_current(burst_context: ComponentContext) -> None:
    learner = build_hyper_cez_learner(burst_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=burst_context.config,
        unroll_steps=burst_context.config.unroll_steps,
        trajectory_size=burst_context.config.trajectory_size,
    )
    _fill_buffer(buffer)

    before_recent = copy.deepcopy(learner._recent_reanalyze_hnets)
    learner.train_step(buffer, skip_reanalyze=True)
    assert not np.allclose(
        np.asarray(
            learner._reanalyze_hnets["dynamics_model"]["hidden_0"]["kernel"]
        ),
        np.asarray(
            learner.train_state["hnets"]["dynamics_model"]["hidden_0"]["kernel"]
        ),
    )
    assert np.allclose(
        np.asarray(
            learner._reanalyze_hnets["dynamics_model"]["hidden_0"]["kernel"]
        ),
        np.asarray(
            before_recent["dynamics_model"]["hidden_0"]["kernel"]
        ),
    )


def test_on_task_boundary_resets_delayed_hnet_copies(burst_context: ComponentContext) -> None:
    learner = build_hyper_cez_learner(burst_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=burst_context.config,
        unroll_steps=burst_context.config.unroll_steps,
        trajectory_size=burst_context.config.trajectory_size,
    )
    _fill_buffer(buffer)
    learner.train_step(buffer, skip_reanalyze=True)
    learner.on_task_boundary(1)
    assert np.allclose(
        np.asarray(
            learner._self_play_hnets["dynamics_model"]["task_embeddings"]
        ),
        np.asarray(
            learner.train_state["hnets"]["dynamics_model"]["task_embeddings"]
        ),
    )


def test_train_burst_after_task_boundary_runs(burst_context: ComponentContext) -> None:
    learner = build_hyper_cez_learner(burst_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=burst_context.config,
        unroll_steps=burst_context.config.unroll_steps,
        trajectory_size=burst_context.config.trajectory_size,
    )
    _fill_buffer(buffer)
    learner.on_task_boundary(1)

    metrics = learner.train_burst(buffer, 4)
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["reg_loss"])
    assert np.isfinite(metrics["cl_beta"])
    assert np.isfinite(metrics["dtheta_norm"])
    assert learner._train_steps == 4


def test_train_burst_task_two_after_second_boundary(burst_context: ComponentContext) -> None:
    learner = build_hyper_cez_learner(burst_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=burst_context.config,
        unroll_steps=burst_context.config.unroll_steps,
        trajectory_size=burst_context.config.trajectory_size,
    )
    _fill_buffer(buffer)
    learner.on_task_boundary(1)
    learner.train_burst(buffer, 2)
    learner.on_task_boundary(2)

    metrics = learner.train_burst(buffer, 4)
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["reg_loss"])
    assert learner._train_steps == 6


def test_train_burst_reanalyze_once_per_sub_burst(burst_context: ComponentContext) -> None:
    config = burst_context.config.with_overrides(
        reanalyze_ratio=1.0,
        gradient_steps_per_rollout=4,
        burst_compile_steps=2,
    )
    context = ComponentContext(
        backend=burst_context.backend,
        config=config,
        env=burst_context.env,
    )
    context.world_model = build_hyper_cez_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    learner = build_hyper_cez_learner(context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    _fill_buffer(buffer)

    with mock.patch(
        "algorl.backends.jax.learners.hypercez.learner.reanalyze_fused_policy_batches",
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
        metrics = learner.train_burst(buffer, 4)

    assert np.isfinite(metrics["loss"])
    assert mock_reanalyze.call_count == 2
