"""EfficientZero training losses (JAX)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn.efficientzero.model import _symexp, _vector_to_scalar

_SUPPORT_EPSILON = 0.001


def _dmc_transform(x: jnp.ndarray) -> jnp.ndarray:
    sign = jnp.where(x < 0.0, -1.0, 1.0)
    return sign * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + _SUPPORT_EPSILON * x


def scalar_to_support(
    targets: jnp.ndarray,
    *,
    support_range: tuple[float, float],
    support_bins: int,
) -> jnp.ndarray:
    """Map scalar targets to a categorical support distribution (DMC preset)."""
    x_min, x_max = support_range
    transformed_min = _dmc_transform(jnp.asarray(x_min, dtype=jnp.float32))
    transformed_max = _dmc_transform(jnp.asarray(x_max, dtype=jnp.float32))
    scale = (transformed_max - transformed_min) / float(support_bins - 1)

    sign = jnp.where(targets < 0.0, -1.0, 1.0)
    transformed = sign * (jnp.sqrt(jnp.abs(targets) + 1.0) - 1.0) + _SUPPORT_EPSILON * targets
    transformed = transformed / scale

    low = transformed_min / scale
    high = transformed_max / scale - 1e-5
    transformed = jnp.clip(transformed, low, high)
    shifted = transformed - low

    low_idx = jnp.floor(shifted).astype(jnp.int32)
    high_idx = jnp.ceil(shifted).astype(jnp.int32)
    weight_high = shifted - jnp.floor(shifted)
    weight_low = 1.0 - weight_high

    batch_shape = targets.shape
    target = jnp.zeros(batch_shape + (support_bins,), dtype=jnp.float32)
    if target.ndim == 1:
        target = target.at[high_idx].add(weight_high)
        target = target.at[low_idx].add(weight_low)
        return target

    flat_target = target.reshape((-1, support_bins))
    flat_high = high_idx.reshape(-1)
    flat_low = low_idx.reshape(-1)
    flat_weight_high = weight_high.reshape(-1)
    flat_weight_low = weight_low.reshape(-1)
    rows = jnp.arange(flat_target.shape[0])
    flat_target = flat_target.at[rows, flat_high].add(flat_weight_high)
    flat_target = flat_target.at[rows, flat_low].add(flat_weight_low)
    return flat_target.reshape(batch_shape + (support_bins,))


def kl_categorical_loss(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(log_probs * target, axis=-1)


def symlog_scalar_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    pred = prediction[..., 0] if prediction.shape[-1] > 1 else prediction
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
    return _vector_to_scalar(values, config.value_support_type, config.support_bins)


def value_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    config: EfficientZeroConfig,
) -> jnp.ndarray:
    if predictions.ndim == 3 and predictions.shape[0] == config.v_num:
        predictions = predictions[0]

    if config.value_support_type == "symlog":
        per_sample = symlog_scalar_loss(predictions, targets)
    else:
        target_support = scalar_to_support(
            targets,
            support_range=config.value_support_range,
            support_bins=config.support_bins,
        )
        per_sample = kl_categorical_loss(predictions, target_support)

    if config.use_IQL:
        reformed = _reduce_value_logits(predictions, config)
        error = reformed - targets
        positive = (error > 0.0).astype(jnp.float32)
        weight = (1.0 - positive) * config.IQL_weight + positive * (1.0 - config.IQL_weight)
        return weight * per_sample
    return per_sample


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
    entropy_samples: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Squashed-Gaussian policy loss (Eq. 6 full pi or Eq. 7 simple pi)."""
    action_dim = policy.shape[-1] // 2
    mean = policy[..., :action_dim]
    std = policy[..., action_dim:]
    clipped_best = jnp.clip(best_action, -0.999, 0.999)

    if (
        candidates is not None
        and target_policy is not None
        and candidates.shape[-2] > 0
        and action_dim == 1
    ):
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
