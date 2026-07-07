"""DMC categorical support encoding and decoding."""

from __future__ import annotations

import jax
import jax.numpy as jnp

_SUPPORT_EPSILON = 0.001


def dmc_transform(x: jnp.ndarray) -> jnp.ndarray:
    """DMC/Gym scalar support transform."""
    sign = jnp.where(x < 0.0, -1.0, 1.0)
    return sign * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + _SUPPORT_EPSILON * x


def dmc_inverse_transform(value: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Inverse of ``dmc_transform`` after expectation in scaled transformed space."""
    sign = jnp.where(value < 0.0, -1.0, 1.0)
    inner = jnp.abs(value) * scale + 1.0 + _SUPPORT_EPSILON
    output = (
        (jnp.sqrt(1.0 + 4.0 * _SUPPORT_EPSILON * inner) - 1.0) / (2.0 * _SUPPORT_EPSILON)
    ) ** 2 - 1.0
    return sign * output


def transformed_bin_centers(
    support_range: tuple[float, float],
    support_bins: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bin centers in transformed space and the per-bin scale divisor."""
    x_min, x_max = support_range
    transformed_min = dmc_transform(jnp.asarray(x_min, dtype=jnp.float32))
    transformed_max = dmc_transform(jnp.asarray(x_max, dtype=jnp.float32))
    scale = (transformed_max - transformed_min) / float(support_bins - 1)
    centers = transformed_min + jnp.arange(support_bins, dtype=jnp.float32) * scale
    return centers, scale


def scalar_to_support(
    targets: jnp.ndarray,
    *,
    support_range: tuple[float, float],
    support_bins: int,
) -> jnp.ndarray:
    """Map scalar targets to a categorical support distribution (DMC preset)."""
    x_min, x_max = support_range
    transformed_min = dmc_transform(jnp.asarray(x_min, dtype=jnp.float32))
    transformed_max = dmc_transform(jnp.asarray(x_max, dtype=jnp.float32))
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


def vector_to_scalar(
    logits: jnp.ndarray,
    *,
    support_type: str,
    support_bins: int,
    support_range: tuple[float, float],
) -> jnp.ndarray:
    """Decode value/reward logits to scalars (EfficientZero-V2 DMC ``vector_to_scalar``)."""
    if support_type not in {"symlog", "support", "discrete"}:
        raise ValueError(f"Unknown support type {support_type!r}")

    if support_type == "symlog":
        sign = jnp.where(logits[..., 0] < 0.0, -1.0, 1.0)
        return sign * (jnp.exp(jnp.abs(logits[..., 0])) - 1.0)

    if logits.shape[-1] < support_bins:
        pad_width = [(0, 0)] * (logits.ndim - 1) + [(0, support_bins - logits.shape[-1])]
        logits = jnp.pad(logits, pad_width, constant_values=-jnp.inf)
    elif logits.shape[-1] > support_bins:
        logits = logits[..., :support_bins]

    probs = jax.nn.softmax(logits, axis=-1)
    centers, scale = transformed_bin_centers(support_range, support_bins)
    value = jnp.sum(probs * centers, axis=-1) / scale
    return dmc_inverse_transform(value, scale)
