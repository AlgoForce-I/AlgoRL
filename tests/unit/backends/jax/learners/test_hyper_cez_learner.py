"""HyperCEZ learner tests."""

from __future__ import annotations

import copy
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest

from algorl.agents.configs import HyperCEZConfig
from algorl.agents.search.hyper_cez import HyperCEZ
from algorl.backends.jax.factory import JAXComponentFactory
from algorl.backends.jax.learners.hypercez import HyperCEZLearner, build_hyper_cez_learner
from algorl.backends.jax.planners.efficientzero import build_efficient_zero_planner
from algorl.backends.jax.world_models.hypercez import (
    HyperCEZWorldModel,
    build_hyper_cez_world_model,
)
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
        gradient_steps_per_rollout=1,
        lr_warm_up=0.0,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    context.world_model = build_hyper_cez_world_model(context)
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


def test_hypercez_learner_registered_as_implemented() -> None:
    factory = JAXComponentFactory()
    assert not factory.is_stub("learner", "hyper_cez")


def test_compose_hypercez_agent(cartpole_training_env: TrainingEnv) -> None:
    config = HyperCEZConfig.for_dmc_state(
        num_tasks=2,
        emb_size=8,
        hnet_arch=(32, 32),
        mcts_simulations=2,
        reanalyze_ratio=0.0,
    )
    agent = HyperCEZ(cartpole_training_env, config=config)
    assert isinstance(agent.world_model, HyperCEZWorldModel)
    assert isinstance(agent.learner, HyperCEZLearner)


def test_train_step_updates_hnets_and_syncs_world_model(
    learner_context: ComponentContext,
) -> None:
    learner = build_hyper_cez_learner(learner_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=learner_context.config,
        unroll_steps=learner_context.config.unroll_steps,
        trajectory_size=learner_context.config.trajectory_size,
    )
    _fill_buffer(buffer)

    before_shared = learner.train_state["shared"]["representation_model"]["LayerNorm_0"]["scale"]
    before_emb = np.array(
        learner.train_state["hnets"]["dynamics_model"]["task_embeddings"][0]
    )
    before_theta = np.array(
        learner.train_state["hnets"]["dynamics_model"]["hidden_0"]["kernel"]
    )
    metrics = learner.train_step(buffer, skip_reanalyze=True)
    after_shared = learner.train_state["shared"]["representation_model"]["LayerNorm_0"]["scale"]
    after_emb = learner.train_state["hnets"]["dynamics_model"]["task_embeddings"][0]
    after_theta = learner.train_state["hnets"]["dynamics_model"]["hidden_0"]["kernel"]

    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])
    assert float(np.max(np.abs(np.asarray(before_shared) - np.asarray(after_shared)))) > 0.0
    assert float(np.max(np.abs(before_emb - np.asarray(after_emb)))) > 0.0
    assert float(np.max(np.abs(before_theta - np.asarray(after_theta)))) > 0.0
    assert learner.world_model.params is not None
    assert learner.world_model.task_id == 0


def test_set_task_id_switches_materialization(learner_context: ComponentContext) -> None:
    learner = build_hyper_cez_learner(learner_context)
    params0 = learner.params
    learner.set_task_id(1)
    assert learner.task_id == 1
    assert learner.world_model.task_id == 1
    # Task embeddings differ => materialized generated leaves differ.
    assert not all(
        jnp.allclose(a, b)
        for a, b in zip(_leaves(params0), _leaves(learner.params), strict=True)
    )


def test_on_task_boundary_snapshots_targets_and_switches_task(
    learner_context: ComponentContext,
) -> None:
    learner = build_hyper_cez_learner(learner_context)
    before = copy.deepcopy(learner.train_state["hnets"])
    learner._task_train_steps = 123
    alpha0 = {
        c: jnp.asarray(learner.train_state["alphas"][0][c] + 0.25)
        for c in learner.config.hnet_components
    }
    alphas = dict(learner.train_state["alphas"])
    alphas[0] = alpha0
    learner.train_state = {**learner.train_state, "alphas": alphas}
    learner.on_task_boundary(1)

    assert learner.task_id == 1
    assert learner._task_train_steps == 0
    assert learner._reg_targets is not None
    assert len(learner._reg_targets["dynamics_model"]) == 1
    assert 0 in learner._shared_snapshots
    for component in learner.config.hnet_components:
        assert np.isclose(
            float(learner.train_state["alphas"][1][component]),
            float(alpha0[component]),
            atol=1e-6,
        )
    assert np.isclose(
        float(
            calc_fix_target_reg_from_learner(learner),
        ),
        0.0,
        atol=1e-5,
    )
    # Snapshot uses pre-boundary weights; switching task does not mutate hnets.
    assert all(
        jnp.allclose(a, b)
        for a, b in zip(
            _leaves(before["dynamics_model"]),
            _leaves(learner.train_state["hnets"]["dynamics_model"]),
            strict=True,
        )
    )


def test_train_step_emits_reg_loss_on_task_one(
    learner_context: ComponentContext,
) -> None:
    learner = build_hyper_cez_learner(learner_context)
    learner.on_task_boundary(1)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=learner_context.config,
        unroll_steps=learner_context.config.unroll_steps,
        trajectory_size=learner_context.config.trajectory_size,
    )
    _fill_buffer(buffer)

    metrics = learner.train_step(buffer, skip_reanalyze=True)
    assert "reg_loss" in metrics
    assert "cl_beta" in metrics
    assert "dtheta_norm" in metrics
    assert np.isfinite(metrics["reg_loss"])
    assert np.isfinite(metrics["dtheta_norm"])


def test_train_step_no_look_ahead_skips_dtheta_metric(
    learner_context: ComponentContext,
) -> None:
    from dataclasses import replace

    learner_context.config = replace(learner_context.config, no_look_ahead=True)
    learner = build_hyper_cez_learner(learner_context)
    learner.on_task_boundary(1)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=learner_context.config,
        unroll_steps=learner_context.config.unroll_steps,
        trajectory_size=learner_context.config.trajectory_size,
    )
    _fill_buffer(buffer)
    metrics = learner.train_step(buffer, skip_reanalyze=True)
    assert "reg_loss" in metrics
    assert metrics.get("dtheta_norm", 0.0) == 0.0


def calc_fix_target_reg_from_learner(learner: HyperCEZLearner) -> float:
    from algorl.backends.jax.nn.hypercez.regularizer import calc_component_reg_loss

    assert learner._reg_targets is not None
    return float(
        calc_component_reg_loss(
            learner.train_state["hnets"],
            hnet_modules=learner.hnet_modules,
            hnet_components=learner.config.hnet_components,
            task_id=learner.task_id,
            reg_targets=learner._reg_targets,
        )
    )


def _leaves(tree):
    import jax

    return jax.tree_util.tree_leaves(tree)


def jax_tree_clone(tree):
    import copy

    return copy.deepcopy(tree)
