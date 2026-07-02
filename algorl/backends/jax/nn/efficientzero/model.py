"""EfficientZero Flax module orchestrator."""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from algorl.agents.configs import EfficientZeroConfig

Params: TypeAlias = chex.ArrayTree

_SUPPORT_MIN = -300.0
_SUPPORT_MAX = 301.0
_SUPPORT_DELTA = 1.0


def _normalize_state(state: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    norm = jnp.linalg.norm(state, axis=-1, keepdims=True)
    return state / (norm + eps)


def _symexp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


def _bin_centers(support_bins: int) -> jnp.ndarray:
    """Support bin centers as a compile-time constant array."""
    step = 600.0 / (support_bins - 1)
    return jnp.array([-300.0 + i * step for i in range(support_bins)], dtype=jnp.float32)


def _support_bins() -> jnp.ndarray:
    count = int(round((_SUPPORT_MAX - _SUPPORT_MIN) / _SUPPORT_DELTA)) + 1
    return _bin_centers(count)


def _pad_support_logits(logits: jnp.ndarray, support_bins: int) -> jnp.ndarray:
    """Pad trailing support dimension to ``support_bins`` for scan-safe reduction."""
    width = support_bins - logits.shape[-1]
    if width <= 0:
        return logits[..., :support_bins]
    pad_spec = [(0, 0)] * (logits.ndim - 1) + [(0, width)]
    return jnp.pad(logits, pad_spec, constant_values=-jnp.inf)


def _vector_to_scalar(logits: jnp.ndarray, support_type: str, support_bins: int) -> jnp.ndarray:
    """Map value/reward logits to a scalar expectation.

    ``support_type`` is fixed per model config, so we branch in Python rather than
    with ``lax.cond`` on the logit width. A traced shape branch here breaks XLA when
    composed with Flax inference inside ``jax.jit``.
    """
    if support_type not in {"symlog", "support", "discrete"}:
        raise ValueError(f"Unknown support type {support_type!r}")

    if support_type == "symlog":
        return _symexp(logits[..., 0])

    padded = _pad_support_logits(logits, support_bins)
    probs = jax.nn.softmax(padded, axis=-1)
    bins = _bin_centers(support_bins)
    return jnp.sum(probs * bins, axis=-1)


class EfficientZero(nn.Module):
    """Composes EfficientZero subnetworks.

    Submodules are injected at construction time. Callers pass a nested ``params``
    dict from ``init`` into the inference methods.
    """

    representation_model: nn.Module
    dynamics_model: nn.Module
    reward_prediction_model: nn.Module
    value_policy_model: nn.Module
    projection_model: nn.Module
    projection_head_model: nn.Module
    config: EfficientZeroConfig = EfficientZeroConfig()

    def _apply(
        self,
        module: nn.Module,
        params: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return module.apply({"params": params}, *args, **kwargs)

    def do_representation(self, params: Params, obs: jnp.ndarray) -> jnp.ndarray:
        state = self._apply(self.representation_model, params["representation_model"], obs)
        if self.config.state_norm:
            state = _normalize_state(state)
        return state

    def do_dynamics(
        self,
        params: Params,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        next_state = self._apply(
            self.dynamics_model,
            params["dynamics_model"],
            state,
            action,
        )
        if self.config.state_norm:
            next_state = _normalize_state(next_state)
        return next_state

    def do_reward_prediction(
        self,
        params: Params,
        next_state: jnp.ndarray,
        reward_hidden: Any = None,
    ) -> tuple[jnp.ndarray, Any]:
        reward_params = params["reward_prediction_model"]
        if self.config.value_prefix:
            value_prefix, reward_hidden = self._apply(
                self.reward_prediction_model,
                reward_params,
                next_state,
                reward_hidden,
            )
            return value_prefix, reward_hidden

        reward = self._apply(self.reward_prediction_model, reward_params, next_state)
        return reward, None

    def do_value_policy_prediction(
        self,
        params: Params,
        state: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._apply(self.value_policy_model, params["value_policy_model"], state)

    def do_projection(
        self,
        params: Params,
        state: jnp.ndarray,
        *,
        with_grad: bool = True,
    ) -> jnp.ndarray:
        proj = self._apply(self.projection_model, params["projection_model"], state)
        if not with_grad:
            return jax.lax.stop_gradient(proj)
        return self._apply(self.projection_head_model, params["projection_head_model"], proj)

    def _reduce_values(
        self,
        values: jnp.ndarray,
        *,
        rng: jax.Array | None,
    ) -> jnp.ndarray:
        if self.config.v_num > 2:
            if rng is None:
                raise ValueError("rng is required when v_num > 2 during inference.")
            indices = jax.random.choice(rng, self.config.v_num, (2,), replace=False)
            values = values[..., indices, :]

        if self.config.value_support_type == "symlog":
            output_values = _symexp(values).min(axis=-2)
        else:
            output_values = jnp.min(
                _vector_to_scalar(
                    values,
                    self.config.value_support_type,
                    self.config.support_bins,
                ),
                axis=-1,
            )

        if self.config.clip_inference_values:
            output_values = jnp.clip(output_values, 0.0, 1e5)
        return output_values

    def _reduce_reward_prefix(self, value_prefix: jnp.ndarray) -> jnp.ndarray:
        if self.config.reward_support_type == "symlog":
            return _symexp(value_prefix)
        return _vector_to_scalar(value_prefix, self.config.reward_support_type, self.config.support_bins)

    def initial_inference(
        self,
        params: Params,
        obs: jnp.ndarray,
        *,
        training: bool = False,
        rng: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        state = self.do_representation(params, obs)
        values, policy = self.do_value_policy_prediction(params, state)
        if training:
            return state, values, policy
        return state, self._reduce_values(values, rng=rng), policy

    def recurrent_inference(
        self,
        params: Params,
        state: jnp.ndarray,
        action: jnp.ndarray,
        reward_hidden: Any,
        *,
        training: bool = False,
        rng: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        next_state = self.do_dynamics(params, state, action)
        value_prefix, reward_hidden = self.do_reward_prediction(
            params,
            next_state,
            reward_hidden,
        )
        values, policy = self.do_value_policy_prediction(params, next_state)
        if training:
            return next_state, value_prefix, values, policy, reward_hidden

        return (
            next_state,
            self._reduce_reward_prefix(value_prefix),
            self._reduce_values(values, rng=rng),
            policy,
            reward_hidden,
        )
