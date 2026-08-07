"""Learner directory checkpoint round-trips."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import jax
import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig, HyperCEZConfig
from algorl.backends.jax.learners.efficientzero import build_efficient_zero_learner
from algorl.backends.jax.learners.hypercez import build_hyper_cez_learner
from algorl.backends.jax.planners.efficientzero import build_efficient_zero_planner
from algorl.backends.jax.world_models.efficientzero import build_efficient_zero_world_model
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


def _fill_buffer(buffer: EfficientZeroReplayBuffer, *, n: int = 12) -> None:
    for index in range(n):
        policy = np.full((4,), 0.25, dtype=np.float32)
        candidates = np.stack(
            [np.asarray([0.1], dtype=np.float32) for _ in range(4)],
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


def _assert_trees_allclose(a: object, b: object, *, rtol: float = 1e-5) -> None:
    leaves_a, struct_a = jax.tree_util.tree_flatten(a)
    leaves_b, struct_b = jax.tree_util.tree_flatten(b)
    assert struct_a == struct_b
    for left, right in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_allclose(
            np.asarray(left),
            np.asarray(right),
            rtol=rtol,
            atol=1e-6,
        )


@pytest.fixture
def ez_context() -> ComponentContext:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        learning_starts=0,
        mcts_simulations=2,
        reanalyze_ratio=0.0,
        use_priority=False,
        burst_compile_steps=1,
    )
    context = ComponentContext(backend=get_backend("jax"), config=config, env=env)
    context.world_model = build_efficient_zero_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    return context


@pytest.fixture
def hyper_context() -> ComponentContext:
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
        burst_compile_steps=1,
        head_init_std=1e-3,
    )
    context = ComponentContext(backend=get_backend("jax"), config=config, env=env)
    context.world_model = build_hyper_cez_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    return context


def test_efficient_zero_learner_checkpoint_roundtrip(
    ez_context: ComponentContext,
    tmp_path: Path,
) -> None:
    learner = build_efficient_zero_learner(ez_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=ez_context.config,
        unroll_steps=2,
        trajectory_size=4,
    )
    _fill_buffer(buffer)
    learner.train_step(buffer, skip_reanalyze=True)
    learner.train_step(buffer, skip_reanalyze=True)

    steps = learner._train_steps
    params_before = jax.tree.map(lambda x: np.array(x), learner.params)
    opt_before = learner._opt_state

    learner.save(tmp_path / "ez_learner")

    restored = build_efficient_zero_learner(ez_context)
    assert restored._train_steps == 0
    restored.load(tmp_path / "ez_learner")

    assert restored._train_steps == steps
    assert restored._obs_running_count == learner._obs_running_count
    _assert_trees_allclose(restored.params, params_before)
    _assert_trees_allclose(restored._opt_state, opt_before)
    # Can continue training after load.
    metrics = restored.train_step(buffer, skip_reanalyze=True)
    assert np.isfinite(metrics["loss"])
    assert restored._train_steps == steps + 1


def test_hypercez_learner_checkpoint_roundtrip(
    hyper_context: ComponentContext,
    tmp_path: Path,
) -> None:
    learner = build_hyper_cez_learner(hyper_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=hyper_context.config,
        unroll_steps=2,
        trajectory_size=4,
    )
    _fill_buffer(buffer)
    learner.train_step(buffer, skip_reanalyze=True)
    learner.on_task_boundary(1)
    learner.train_step(buffer, skip_reanalyze=True)

    assert learner.task_id == 1
    assert learner._reg_targets is not None
    steps = learner._train_steps
    task_steps = learner._task_train_steps
    train_state_before = jax.tree.map(lambda x: np.array(x), learner.train_state)

    learner.save(tmp_path / "hyper_learner")

    restored = build_hyper_cez_learner(hyper_context)
    restored.load(tmp_path / "hyper_learner")

    assert restored.task_id == 1
    assert restored._train_steps == steps
    assert restored._task_train_steps == task_steps
    assert restored._reg_targets is not None
    _assert_trees_allclose(restored.train_state, train_state_before)
    metrics = restored.train_step(buffer, skip_reanalyze=True)
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["reg_loss"])
