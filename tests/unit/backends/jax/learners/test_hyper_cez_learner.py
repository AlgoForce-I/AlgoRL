"""HyperCEZ learner tests."""

from __future__ import annotations

import copy
from unittest import mock

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorl.agents.configs import HyperCEZConfig
from algorl.agents.search.hyper_cez import HyperCEZ
from algorl.backends.jax.factory import JAXComponentFactory
from algorl.backends.jax.learners.hypercez import HyperCEZLearner, build_hyper_cez_learner
from algorl.backends.jax.learners.hypercez.learner import _clip_tree_by_global_norm
from algorl.backends.jax.nn.efficientzero.obs_norm import INITIAL_OBS_RUNNING_COUNT
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
        head_init_std=1e-3,
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
        head_init_std=1e-3,
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
    before_base = np.array(
        learner.train_state["base"]["dynamics_model"]["Dense_0"]["kernel"]
    )
    metrics = learner.train_step(buffer, skip_reanalyze=True)
    after_shared = learner.train_state["shared"]["representation_model"]["LayerNorm_0"]["scale"]
    after_emb = learner.train_state["hnets"]["dynamics_model"]["task_embeddings"][0]
    after_theta = learner.train_state["hnets"]["dynamics_model"]["hidden_0"]["kernel"]
    after_base = learner.train_state["base"]["dynamics_model"]["Dense_0"]["kernel"]

    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])
    assert float(np.max(np.abs(np.asarray(before_shared) - np.asarray(after_shared)))) > 0.0
    assert float(np.max(np.abs(before_emb - np.asarray(after_emb)))) > 0.0
    assert float(np.max(np.abs(before_theta - np.asarray(after_theta)))) > 0.0
    # Default frozen_base_weights=True ⇒ W0 must not move.
    assert float(np.max(np.abs(before_base - np.asarray(after_base)))) == 0.0
    assert learner.world_model.params is not None
    assert learner.world_model.task_id == 0


def test_unfrozen_base_weights_update_slowly(
    cartpole_training_env: TrainingEnv,
) -> None:
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
        frozen_base_weights=False,
        lr_hyper=3e-4,
        lr_main_to_lr_hyper_ratio=50.0,
        head_init_std=1e-3,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    context.world_model = build_hyper_cez_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    learner = build_hyper_cez_learner(context)
    assert learner.config.frozen_base_weights is False
    assert np.isclose(learner._base_learning_rate(), 3e-4 / 50.0)

    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    _fill_buffer(buffer)

    before_base = np.array(
        learner.train_state["base"]["dynamics_model"]["Dense_0"]["kernel"]
    )
    learner.train_step(buffer, skip_reanalyze=True)
    after_base = np.array(
        learner.train_state["base"]["dynamics_model"]["Dense_0"]["kernel"]
    )
    assert float(np.max(np.abs(before_base - after_base))) > 0.0
    # World-model W0 snapshot tracks the trainable base.
    wm_gen = learner.world_model.frozen_ez["dynamics_model"]["Dense_0"]["kernel"]
    assert np.allclose(np.asarray(wm_gen), after_base)


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


def test_on_task_boundary_resets_live_obs_norm_and_keeps_snapshot(
    learner_context: ComponentContext,
) -> None:
    learner = build_hyper_cez_learner(learner_context)
    shared = dict(learner.train_state["shared"])
    rep = dict(shared["representation_model"])
    poisoned_var = jnp.zeros_like(rep["running_var"])
    poisoned_mean = jnp.ones_like(rep["running_mean"])
    rep["running_var"] = poisoned_var
    rep["running_mean"] = poisoned_mean
    shared["representation_model"] = rep
    learner.train_state = {**learner.train_state, "shared": shared}
    learner._obs_running_count = 1_827_406_824

    learner.on_task_boundary(1)

    live_rep = learner.train_state["shared"]["representation_model"]
    np.testing.assert_allclose(np.asarray(live_rep["running_mean"]), 0.0)
    np.testing.assert_allclose(np.asarray(live_rep["running_var"]), 1.0)
    assert learner._obs_running_count == INITIAL_OBS_RUNNING_COUNT
    snapped = learner._shared_snapshots[0]["representation_model"]
    np.testing.assert_allclose(np.asarray(snapped["running_mean"]), 1.0)
    np.testing.assert_allclose(np.asarray(snapped["running_var"]), 0.0)


def test_non_finite_loss_does_not_update_hnets_via_reg(
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
    hnets_before = copy.deepcopy(jax.tree.map(np.asarray, learner.train_state["hnets"]))

    with mock.patch(
        "algorl.backends.jax.learners.hypercez.learner._loss_from_batch",
        return_value=(
            jnp.asarray(float("nan"), dtype=jnp.float32),
            {"priorities": jnp.full((2,), float("nan"), dtype=jnp.float32)},
        ),
    ):
        learner._recompile_train_kernels()
        metrics = learner.train_step(buffer, skip_reanalyze=True)

    assert not np.isfinite(metrics["loss"])
    hnets_after = jax.tree.map(np.asarray, learner.train_state["hnets"])
    assert all(
        np.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(hnets_before),
            jax.tree.leaves(hnets_after),
            strict=True,
        )
    )


def test_lookahead_clip_zeros_non_finite_grads() -> None:
    clipped = _clip_tree_by_global_norm(
        {"g": jnp.asarray([jnp.inf, jnp.nan, 1.0], dtype=jnp.float32)},
        5.0,
    )
    assert np.all(np.isfinite(np.asarray(clipped["g"])))


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


def _nullspace_context(env: TrainingEnv, **overrides: object) -> ComponentContext:
    fields = dict(
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
        head_init_std=1e-3,
        cl_strategy="nullspace",
    )
    fields.update(overrides)
    context = ComponentContext(
        backend=get_backend("jax"),
        config=HyperCEZConfig.for_dmc_state(**fields),
        env=env,
    )
    context.world_model = build_hyper_cez_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    return context


def _hnet_outputs(learner: HyperCEZLearner, task_id: int) -> np.ndarray:
    from algorl.backends.jax.nn.hypercez.hyper_model import apply_hypernetwork

    return np.concatenate(
        [
            np.ravel(np.asarray(leaf))
            for component in learner.config.hnet_components
            for leaf in apply_hypernetwork(
                learner.hnet_modules[component],
                learner.train_state["hnets"][component],
                task_id,
            )
        ]
    )


def _filled_buffer(config: HyperCEZConfig) -> EfficientZeroReplayBuffer:
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=config.unroll_steps,
        trajectory_size=config.trajectory_size,
    )
    _fill_buffer(buffer)
    return buffer


def test_nullspace_keeps_previous_task_exactly_while_training_current(
    cartpole_training_env: TrainingEnv,
) -> None:
    context = _nullspace_context(cartpole_training_env)
    learner = build_hyper_cez_learner(context)
    buffer = _filled_buffer(context.config)
    for _ in range(3):
        learner.train_step(buffer, skip_reanalyze=True)
    learner.on_task_boundary(1)
    assert learner._nullspace_bases is not None
    assert learner._defer_theta() is False

    task0_hnet = _hnet_outputs(learner, 0)
    task1_hnet = _hnet_outputs(learner, 1)
    task0_params = copy.deepcopy(
        {c: learner.materialize_task(0)[c] for c in learner.config.hnet_components}
    )
    alpha0 = copy.deepcopy(learner.train_state["alphas"][0])
    for _ in range(5):
        metrics = learner.train_step(buffer, skip_reanalyze=True)

    assert metrics["cl_beta"] == 0.0
    np.testing.assert_allclose(_hnet_outputs(learner, 0), task0_hnet, atol=1e-5)
    assert np.linalg.norm(_hnet_outputs(learner, 1) - task1_hnet) > 0.0
    task0_after = {c: learner.materialize_task(0)[c] for c in learner.config.hnet_components}
    for before, after in zip(_leaves(task0_params), _leaves(task0_after), strict=True):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=1e-5)
    for component in learner.config.hnet_components:
        assert float(learner.train_state["alphas"][0][component]) == float(alpha0[component])

    retention = learner.retention_target_metrics()
    assert retention["retention/fix_target_reg"] < 1e-8
    assert 0.0 < retention["retention/free_frac/dynamics_model"] <= 1.0


def test_nullspace_burst_keeps_previous_tasks(cartpole_training_env: TrainingEnv) -> None:
    context = _nullspace_context(
        cartpole_training_env, burst_compile_steps=2, gradient_steps_per_rollout=4
    )
    learner = build_hyper_cez_learner(context)
    buffer = _filled_buffer(context.config)
    learner.train_burst(buffer, 2)
    learner.on_task_boundary(1)
    learner.train_burst(buffer, 2)
    learner.on_task_boundary(2)
    before = {task: _hnet_outputs(learner, task) for task in (0, 1)}

    metrics = learner.train_burst(buffer, 4)

    assert np.isfinite(metrics["loss"])
    for task in (0, 1):
        np.testing.assert_allclose(_hnet_outputs(learner, task), before[task], atol=1e-5)


def test_nullspace_bases_rebuilt_after_checkpoint_load(
    cartpole_training_env: TrainingEnv,
    tmp_path,
) -> None:
    context = _nullspace_context(cartpole_training_env)
    learner = build_hyper_cez_learner(context)
    buffer = _filled_buffer(context.config)
    learner.train_step(buffer, skip_reanalyze=True)
    learner.on_task_boundary(1)
    learner.save(tmp_path / "learner")

    restored = build_hyper_cez_learner(_nullspace_context(cartpole_training_env))
    restored.load(tmp_path / "learner")
    assert restored.task_id == 1
    assert restored._protected_task_ids() == [0]
    assert restored._nullspace_bases is not None
    task0 = _hnet_outputs(restored, 0)
    restored.train_step(buffer, skip_reanalyze=True)
    np.testing.assert_allclose(_hnet_outputs(restored, 0), task0, atol=1e-5)


def _balanced_context(env: TrainingEnv, **overrides: object) -> ComponentContext:
    return _nullspace_context(env, **{"cl_strategy": "fix_target", **overrides})


def test_mix_component_grads_removes_conflict_and_scales_reg() -> None:
    from algorl.backends.jax.learners.hypercez.learner import _mix_component_grads

    task = {"w": jnp.asarray([1.0, 1.0])}
    unset = jnp.asarray(0.0)
    # Conflict (dot < 0): task loses its component along reg, then reg is scaled
    # to λ times the remaining task norm.
    mixed, diagnostics = _mix_component_grads(
        task,
        {"w": jnp.asarray([-1.0, 0.0])},
        reg_lambda=jnp.asarray(2.0),
        task_norm_ref=unset,
        reg_norm_ref=unset,
        conflict_projection=True,
    )
    np.testing.assert_allclose(np.asarray(mixed["w"]), [-2.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(float(diagnostics["scale"]), 2.0, atol=1e-6)
    assert float(diagnostics["conflict_cos"]) < 0.0

    # No conflict: task gradient is untouched.
    aligned, diagnostics = _mix_component_grads(
        task,
        {"w": jnp.asarray([1.0, 0.0])},
        reg_lambda=jnp.asarray(1.0),
        task_norm_ref=unset,
        reg_norm_ref=unset,
        conflict_projection=True,
    )
    np.testing.assert_allclose(np.asarray(aligned["w"]), [1.0 + np.sqrt(2.0), 1.0], atol=1e-5)

    # Host EMAs replace per-step norms once set.
    _, diagnostics = _mix_component_grads(
        task,
        {"w": jnp.asarray([1.0, 0.0])},
        reg_lambda=jnp.asarray(1.0),
        task_norm_ref=jnp.asarray(3.0),
        reg_norm_ref=jnp.asarray(6.0),
        conflict_projection=True,
    )
    np.testing.assert_allclose(float(diagnostics["scale"]), 0.5, atol=1e-6)


def test_balanced_fix_target_reports_diagnostics_and_updates_norm_emas(
    cartpole_training_env: TrainingEnv,
) -> None:
    context = _balanced_context(cartpole_training_env)
    learner = build_hyper_cez_learner(context)
    learner.on_task_boundary(1)
    assert learner._defer_theta()

    metrics = learner.train_step(_filled_buffer(context.config), skip_reanalyze=True)

    for component in context.config.hnet_components:
        assert np.isfinite(metrics[f"cl_lambda/{component}"])
        assert 0.0 <= metrics[f"cl_task_share/{component}"] <= 1.0 + 1e-5
        assert -1.0 - 1e-5 <= metrics[f"cl_conflict_cos/{component}"] <= 1.0 + 1e-5
    assert not any(key.startswith(("cl_task_grad_norm/", "cl_reg_grad_norm/")) for key in metrics)
    assert np.all(np.asarray(learner._reg_balance_state["task_norm"]) > 0.0)
    assert np.all(np.asarray(learner._reg_balance_state["reg_norm"]) > 0.0)


def test_reg_lambda_controller_tracks_drift_budget(cartpole_training_env: TrainingEnv) -> None:
    context = _balanced_context(
        cartpole_training_env,
        reg_balance_interval=10,
        reg_drift_budget=1e-3,
        reg_lambda_init=10.0,
        reg_lambda_min=1.0,
        reg_lambda_max=100.0,
        reg_task_share_floor=0.0,  # exercise the drift loop on its own
    )
    learner = build_hyper_cez_learner(context)
    learner.on_task_boundary(1)
    count = len(context.config.hnet_components)

    def lambdas() -> np.ndarray:
        return np.asarray(learner._reg_balance_state["lambda"])

    def drift(value: float):
        return mock.patch.object(learner, "_relative_reg_drift", return_value=np.full(count, value))

    with drift(1e-1):
        learner._maybe_update_reg_lambda(5, 8)  # interval not crossed: no measurement
    np.testing.assert_allclose(lambdas(), 10.0)
    with drift(2e-3):  # 2x over budget, first measurement → tighten 2x
        learner._maybe_update_reg_lambda(8, 10)
    np.testing.assert_allclose(lambdas(), 20.0)
    with drift(1.0):  # far over and rising → capped step
        learner._maybe_update_reg_lambda(10, 20)
    np.testing.assert_allclose(lambdas(), 40.0)
    for step in range(2, 8):  # same level: smoothed drift catches up, keeps tightening
        with drift(1.0):
            learner._maybe_update_reg_lambda(step * 10, step * 10 + 10)
    np.testing.assert_allclose(lambdas(), 100.0)  # hard-over guard drives it to the max
    # Settles just over budget and stays there: the smoothed drift catches up and
    # λ decays instead of holding, since tightening is not what keeps it flat.
    trace = []
    for step in range(8, 30):
        with drift(1e-3 * 1.05):
            learner._maybe_update_reg_lambda(step * 10, step * 10 + 10)
        trace.append(float(lambdas()[0]))
    assert trace[-1] < trace[-5] < 100.0, trace
    assert learner._reg_lambda_history["held"].all()
    before = trace[-1]
    for step in range(30, 40):  # drift gone → loosen toward the budget
        with drift(0.0):
            learner._maybe_update_reg_lambda(step * 10, step * 10 + 10)
    assert float(lambdas()[0]) < before * 0.5

    # A new task keeps λ as a prior but resets the norm EMAs and controller history.
    carried = lambdas().copy()
    learner.on_task_boundary(2)
    np.testing.assert_allclose(lambdas(), carried, rtol=1e-5)
    assert np.all(np.asarray(learner._reg_balance_state["task_norm"]) == 0.0)
    assert learner._reg_lambda_history["drift"] is None
    assert learner._reg_lambda_history["ref"] is None
    assert np.all(np.isnan(learner._reg_share_ema))


def test_next_reg_lambda_stops_tightening_against_a_flat_drift_floor() -> None:
    """A drift that sits above budget without growing must not ratchet λ."""
    from algorl.backends.jax.learners.hypercez.learner import _next_reg_lambda

    kw = dict(budget=1e-3, lambda_min=0.1, lambda_max=1e3, share_floor=0.0)
    lam = np.array([1.0])
    ema = ref = None
    rng = np.random.default_rng(0)
    peak = 0.0
    for _ in range(400):
        # 3x over budget, fluctuating like the measured jitter floor.
        drift = np.array([3e-3 * float(np.exp(rng.normal(0.0, 0.35)))])
        lam, _, _, ema, ref = _next_reg_lambda(lam, drift, ema, ref, None, **kw)
        peak = max(peak, float(lam[0]))
    assert peak < 20.0, f"lambda ratcheted to {peak} on a flat, over-budget drift"
    assert float(lam[0]) < 1.0

    # Genuine growth still tightens, fast.
    lam, ema, ref = np.array([1.0]), None, None
    for value in np.geomspace(1e-3, 1.0, 12):
        lam, _, _, ema, ref = _next_reg_lambda(lam, np.array([value]), ema, ref, None, **kw)
    assert float(lam[0]) > 100.0

    nan_lam, _, _, _, _ = _next_reg_lambda(
        np.array([10.0]), np.array([np.nan]), None, None, None, **kw
    )
    assert nan_lam[0] == 5.0  # non-finite drift reads as 0 → under budget → loosen


def test_next_reg_lambda_keeps_the_task_share_floor() -> None:
    """λ is capped so the new task always keeps a usable part of its gradient."""
    from algorl.backends.jax.learners.hypercez.learner import (
        _lambda_share_cap,
        _next_reg_lambda,
    )

    # Worst case ||g_mix|| = ||g_task|| sqrt(1 + λ²): the cap inverts that bound.
    assert _lambda_share_cap(0.2, 1e3) == pytest.approx(np.sqrt(1 / 0.04 - 1))
    assert _lambda_share_cap(0.0, 1e3) == 1e3  # disabled

    kw = dict(budget=1e-9, lambda_min=0.1, lambda_max=1e3, share_floor=0.2)
    lam, ema, ref = np.array([900.0]), None, None
    for _ in range(3):  # drift far over budget: the old loop pinned λ at the max
        lam, _, _, ema, ref = _next_reg_lambda(lam, np.array([1e-3]), ema, ref, None, **kw)
    assert float(lam[0]) <= _lambda_share_cap(0.2, 1e3) + 1e-9

    # A measured share under the floor shrinks λ further, even at the cap.
    lam = np.array([4.0])
    for _ in range(4):
        lam, _, _, ema, ref = _next_reg_lambda(
            lam, np.array([1e-3]), ema, ref, np.array([0.05]), **kw
        )
    assert float(lam[0]) < 1.0


def test_reg_lambda_persists_through_checkpoint(
    cartpole_training_env: TrainingEnv,
    tmp_path,
) -> None:
    learner = build_hyper_cez_learner(_balanced_context(cartpole_training_env))
    learner.on_task_boundary(1)
    from algorl.backends.jax.learners.hypercez.learner import _lambda_share_cap

    config = learner.config
    cap = _lambda_share_cap(config.reg_task_share_floor, config.reg_lambda_max)
    saved = [1.0, 2.0, 3.0, 11.0]  # the last one is above the share-floor cap
    learner._reg_balance_state = {
        **learner._reg_balance_state,
        "lambda": jnp.asarray(saved, dtype=jnp.float32),
    }
    learner.save(tmp_path / "learner")

    restored = build_hyper_cez_learner(_balanced_context(cartpole_training_env))
    restored.load(tmp_path / "learner")
    # λ survives the round trip, but the live config's plasticity floor still
    # applies: a λ saved before the floor existed cannot come back above it.
    np.testing.assert_allclose(
        np.asarray(restored._reg_balance_state["lambda"]),
        np.minimum(saved, cap),
        rtol=1e-6,
    )
    assert restored._jit_component_reg is not None
    assert restored._relative_reg_drift().shape == (4,)


def test_loss_ratio_balance_keeps_original_beta(cartpole_training_env: TrainingEnv) -> None:
    context = _balanced_context(cartpole_training_env, reg_balance="loss_ratio")
    learner = build_hyper_cez_learner(context)
    learner.on_task_boundary(1)
    metrics = learner.train_step(_filled_buffer(context.config), skip_reanalyze=True)
    assert metrics["cl_beta"] > 0.0
    assert metrics["cl_lambda/dynamics_model"] == 0.0
    assert learner._jit_component_reg is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"reg_balance": "bogus"},
        {"reg_drift_budget": 0.0},
        {"reg_lambda_init": 0.01},
    ],
)
def test_fix_target_rejects_invalid_balance_configs(
    cartpole_training_env: TrainingEnv,
    overrides: dict,
) -> None:
    with pytest.raises(ValueError):
        build_hyper_cez_learner(_balanced_context(cartpole_training_env, **overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"hnet_type": "chunked", "chunk_dim": 64, "cemb_size": 8},
        {"frozen_base_weights": False},
        {"plastic_prev_tembs": True},
        {"cl_strategy": "bogus"},
    ],
)
def test_nullspace_rejects_incompatible_configs(
    cartpole_training_env: TrainingEnv,
    overrides: dict,
) -> None:
    with pytest.raises(ValueError):
        build_hyper_cez_learner(_nullspace_context(cartpole_training_env, **overrides))
