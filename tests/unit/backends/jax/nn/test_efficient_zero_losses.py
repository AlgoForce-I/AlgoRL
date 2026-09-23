"""EfficientZero loss parity tests."""

from __future__ import annotations

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.learners.efficientzero.learner import _loss_from_batch
from algorl.backends.jax.nn.efficientzero.build import (
    build_efficient_zero_model_from_env,
    init_efficient_zero_params_from_env,
)
from algorl.backends.jax.nn.efficientzero.losses import (
    continuous_policy_loss,
    squashed_normal_log_prob,
    symlog_scalar_loss,
    value_loss,
)
from algorl.envs.training_env import TrainingEnv


def test_continuous_policy_loss_multi_dim_uses_the_improved_target() -> None:
    """Eq. 6 applies at every action_dim; Eq. 7 stays available as a fallback."""
    policy = jnp.concatenate(
        [
            jnp.zeros((2, 6), dtype=jnp.float32),
            jnp.full((2, 6), 0.5, dtype=jnp.float32),
        ],
        axis=-1,
    )
    best_action = jnp.full((2, 6), 0.2, dtype=jnp.float32)
    candidates = jnp.asarray(
        np.random.default_rng(0).uniform(-0.6, 0.6, size=(2, 8, 6)), dtype=jnp.float32
    )
    target_policy = jnp.full((2, 8), 1.0 / 8.0, dtype=jnp.float32)

    improved, entropy = continuous_policy_loss(
        policy,
        best_action,
        candidates=candidates,
        target_policy=target_policy,
        entropy_rng=jax.random.PRNGKey(0),
    )
    simple, _ = continuous_policy_loss(
        policy,
        best_action,
        candidates=candidates,
        target_policy=target_policy,
        entropy_rng=jax.random.PRNGKey(0),
        use_improved_target=False,
    )
    expected_simple = -squashed_normal_log_prob(
        policy[..., :6], policy[..., 6:], jnp.clip(best_action, -0.999, 0.999)
    )

    assert np.all(np.isfinite(improved))
    assert np.all(np.isfinite(entropy))
    assert np.all(entropy > 0.0)
    # Eq. 7 is exactly the single-action likelihood, Eq. 6 is not.
    assert np.allclose(simple, expected_simple, atol=1e-5)
    assert not np.allclose(improved, simple, atol=1e-3)

    # Weights that do not index the candidates fall back instead of mis-pairing.
    mismatched, _ = continuous_policy_loss(
        policy,
        best_action,
        candidates=candidates,
        target_policy=target_policy[..., :4],
        entropy_rng=jax.random.PRNGKey(0),
    )
    assert np.allclose(mismatched, expected_simple, atol=1e-5)


def test_improved_target_widens_a_collapsed_policy() -> None:
    """The reason for Eq. 6: it has a gradient that restores action spread."""
    action_dim = 4
    # Collapsed onto the action-box corner, as the push-back run ended up.
    mean = jnp.full((action_dim,), 2.0, dtype=jnp.float32)
    std = jnp.full((action_dim,), 0.1, dtype=jnp.float32)
    policy = jnp.concatenate([mean, std])
    candidates = jnp.asarray(
        np.random.default_rng(1).uniform(-0.8, 0.8, size=(12, action_dim)), dtype=jnp.float32
    )
    target_policy = jnp.full((12,), 1.0 / 12.0, dtype=jnp.float32)
    best_action = candidates[0]

    def improved(p):
        loss, _ = continuous_policy_loss(p, best_action, candidates=candidates, target_policy=target_policy)
        return jnp.sum(loss)

    def simple(p):
        loss, _ = continuous_policy_loss(p, best_action, use_improved_target=False)
        return jnp.sum(loss)

    # Descending the improved loss raises std and pulls the mean back inside.
    grad_improved = jax.grad(improved)(policy)
    assert float(grad_improved[action_dim]) < 0.0, "improved target should widen std"
    assert float(grad_improved[0]) > 0.0, "improved target should pull the mean inward"
    grad_simple = jax.grad(simple)(policy)
    assert float(grad_simple[action_dim]) != 0.0


def test_value_loss_applies_iql_weight_when_disabled() -> None:
    config = EfficientZeroConfig.for_dmc_state(
        require_implemented=False,
        use_IQL=False,
        value_support_type="symlog",
        v_num=1,
    )
    predictions = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    targets = jnp.array([1.0, 0.0], dtype=jnp.float32)

    weighted = value_loss(predictions, targets, config)
    base = symlog_scalar_loss(predictions, targets)

    assert np.allclose(weighted, 0.5 * base)


def test_loss_from_batch_runs_for_continuous_control() -> None:
    env = TrainingEnv.from_gymnasium(gym.make("HalfCheetah-v5"))
    config = EfficientZeroConfig.for_dmc_state(
        require_implemented=False,
        batch_size=2,
        unroll_steps=1,
        trajectory_size=4,
    )
    model = build_efficient_zero_model_from_env(config, env)
    params = init_efficient_zero_params_from_env(
        model,
        jax.random.PRNGKey(0),
        env,
    )
    batch_size = 2
    window = config.unroll_steps + 1
    action_dim = env.action_dim

    obs_dim = env.observation_shape if isinstance(env.observation_shape, int) else env.observation_shape[0]

    batch = {
        "observations": jnp.zeros((batch_size, window, obs_dim), dtype=jnp.float32),
        "actions": jnp.zeros((batch_size, config.unroll_steps, action_dim), dtype=jnp.float32),
        "rewards": jnp.zeros((batch_size, config.unroll_steps), dtype=jnp.float32),
        "policy_targets": jnp.full((batch_size, window, 8), 1.0 / 8.0, dtype=jnp.float32),
        "value_targets": jnp.zeros((batch_size, window), dtype=jnp.float32),
        "policy_candidates": jnp.zeros((batch_size, window, 8, action_dim), dtype=jnp.float32),
        "best_actions": jnp.full((batch_size, window, action_dim), 0.1, dtype=jnp.float32),
        "dones": jnp.zeros((batch_size, config.unroll_steps), dtype=jnp.bool_),
        "masks": jnp.ones((batch_size, config.unroll_steps), dtype=jnp.float32),
        "weights": jnp.ones((batch_size,), dtype=jnp.float32),
        "indices": jnp.zeros((batch_size,), dtype=jnp.int32),
    }

    loss, metrics = _loss_from_batch(
        params,
        batch,
        model=model,
        config=config,
        rng=jax.random.PRNGKey(1),
    )

    assert np.isfinite(float(loss))
    # With the improved target the policy term is a cross-entropy against a
    # continuous density, so it may be negative; only finiteness is meaningful.
    assert np.isfinite(float(metrics["policy_loss"]))
    assert float(metrics["entropy"]) > 0.0
    assert metrics["priorities"].shape == (batch_size,)

    # Regression: the loss must respond to non-first samples in the batch
    # (the vector networks used to silently collapse the batch dimension).
    perturbed = dict(batch)
    observations = np.asarray(batch["observations"]).copy()
    observations[1] += 1.0
    perturbed["observations"] = jnp.asarray(observations, dtype=jnp.float32)
    perturbed_loss, _ = _loss_from_batch(
        params,
        perturbed,
        model=model,
        config=config,
        rng=jax.random.PRNGKey(1),
    )
    assert abs(float(loss) - float(perturbed_loss)) > 1e-8
