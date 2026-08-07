"""EfficientZero training losses (JAX)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn.efficientzero.model import _symexp
from algorl.backends.jax.nn.efficientzero.support import scalar_to_support, vector_to_scalar


def kl_categorical_loss(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(log_probs * target, axis=-1)


def symlog_scalar_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    pred = prediction
    if pred.ndim == target.ndim + 1:
        # Scalar heads emit a trailing singleton bin.
        pred = pred[..., 0]
    target_symlog = jnp.sign(target) * jnp.log1p(jnp.abs(target))
    return 0.5 * (pred - target_symlog) ** 2


def cosine_consistency_loss(projected: jnp.ndarray, target_projected: jnp.ndarray) -> jnp.ndarray:
    left = projected / (jnp.linalg.norm(projected, axis=-1, keepdims=True) + 1e-5)
    right = target_projected / (jnp.linalg.norm(target_projected, axis=-1, keepdims=True) + 1e-5)
    return -jnp.sum(left * right, axis=-1)


def squashed_normal_log_prob(mean: jnp.ndarray, std: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    clipped = jnp.clip(actions, -0.999, 0.999)
    pre_tanh = jnp.arctanh(clipped)
    safe_std = jnp.maximum(std, 1e-6)
    var = safe_std**2
    log_gaussian = -0.5 * jnp.sum(
        ((pre_tanh - mean) ** 2) / var + 2.0 * jnp.log(safe_std) + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )
    log_det = jnp.sum(jnp.log(1.0 - jnp.tanh(pre_tanh) ** 2 + 1e-6), axis=-1)
    return log_gaussian - log_det


def _reduce_value_logits(values: jnp.ndarray, config: EfficientZeroConfig) -> jnp.ndarray:
    if config.value_support_type == "symlog":
        return _symexp(values[..., 0])
    return vector_to_scalar(
        values,
        support_type=config.value_support_type,
        support_bins=config.support_bins,
        support_range=config.value_support_range,
    )


def value_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    config: EfficientZeroConfig,
) -> jnp.ndarray:
    """Value loss over all ensemble heads with per-head IQL weighting."""
    has_ensemble_axis = predictions.ndim == targets.ndim + 2
    if not has_ensemble_axis:
        predictions = predictions[None, ...]
    ensemble_targets = jnp.broadcast_to(
        targets,
        (predictions.shape[0],) + targets.shape,
    )

    if config.value_support_type == "symlog":
        per_sample = symlog_scalar_loss(predictions, ensemble_targets)
    else:
        target_support = scalar_to_support(
            ensemble_targets,
            support_range=config.value_support_range,
            support_bins=config.support_bins,
        )
        per_sample = kl_categorical_loss(predictions, target_support)

    # EfficientZero-V2 Value_loss applies IQL_weight=0.5 asymmetry even when use_IQL=False.
    iql_weight = config.IQL_weight if config.use_IQL else 0.5
    reformed = _reduce_value_logits(predictions, config)
    error = reformed - ensemble_targets
    positive = (error > 0.0).astype(jnp.float32)
    weight = (1.0 - positive) * iql_weight + positive * (1.0 - iql_weight)
    return jnp.mean(weight * per_sample, axis=0)


def reward_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    config: EfficientZeroConfig,
) -> jnp.ndarray:
    if config.reward_support_type == "symlog":
        return symlog_scalar_loss(predictions, targets)
    target_support = scalar_to_support(
        targets,
        support_range=config.reward_support_range,
        support_bins=config.support_bins,
    )
    return kl_categorical_loss(predictions, target_support)


def continuous_policy_loss(
    policy: jnp.ndarray,
    best_action: jnp.ndarray,
    *,
    candidates: jnp.ndarray | None = None,
    target_policy: jnp.ndarray | None = None,
    entropy_rng: jax.Array | None = None,
    entropy_samples: int = 1024,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Squashed-Gaussian policy loss (Eq. 6 full pi or Eq. 7 simple pi)."""
    action_dim = policy.shape[-1] // 2
    mean = policy[..., :action_dim]
    std = policy[..., action_dim:]
    clipped_best = jnp.clip(best_action, -0.999, 0.999)

    use_full_pi = (
        action_dim == 1
        and candidates is not None
        and target_policy is not None
        and candidates.shape[-2] > 0
    )
    if use_full_pi:
        log_probs = squashed_normal_log_prob(
            mean[..., None, :],
            std[..., None, :],
            jnp.clip(candidates, -0.999, 0.999),
        )
        policy_loss = -jnp.sum(target_policy * log_probs, axis=-1)
    else:
        policy_loss = -squashed_normal_log_prob(mean, std, clipped_best)

    if entropy_rng is None:
        return policy_loss, jnp.zeros_like(policy_loss)

    keys = jax.random.split(entropy_rng, entropy_samples)
    eps = jax.vmap(lambda key: jax.random.normal(key, mean.shape))(keys)
    sampled = jnp.tanh(mean[None, ...] + std[None, ...] * eps)
    sampled = jnp.clip(sampled, -0.999, 0.999)
    log_probs = jax.vmap(
        lambda actions: squashed_normal_log_prob(mean, std, actions),
        in_axes=0,
    )(sampled)
    entropy = -jnp.mean(log_probs, axis=0)
    return policy_loss, entropy


def apply_half_gradient(state: jnp.ndarray) -> jnp.ndarray:
    stopped = jax.lax.stop_gradient(state)
    return stopped + 0.5 * (state - stopped)
