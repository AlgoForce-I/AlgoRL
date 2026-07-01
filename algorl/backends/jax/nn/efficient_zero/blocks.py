"""Flax building blocks for EfficientZero subnetworks (HyperCEZ-aligned)."""

from __future__ import annotations

import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def support_output_size(support_type: str, support_bins: int = 51) -> int:
    if support_type == "symlog":
        return 1
    if support_type in {"support", "discrete"}:
        return support_bins
    raise ValueError(f"Unknown support type {support_type!r}")


def zero_init_dense(features: int, name: str | None = None) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.zeros,
        name=name,
    )


def _layer_sizes(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


class HyperMLP(nn.Module):
    """MLP with optional BatchNorm-style LayerNorm and zero last layer."""

    features: Sequence[int]
    use_layernorm: bool = True
    zero_last: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for index, width in enumerate(self.features):
            is_last = index == len(self.features) - 1
            if is_last and self.zero_last:
                x = zero_init_dense(width)(x)
            else:
                x = nn.Dense(width)(x)
            if self.use_layernorm and not is_last:
                x = nn.LayerNorm()(x)
            if not is_last:
                x = nn.relu(x)
        return x


class ImproveResidualBlock(nn.Module):
    """Pre-LN residual block from HyperCEZ ``alt_model``."""

    hidden_shape: int
    block_shape: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.block_shape)(y)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_shape)(y)
        return residual + y


def _to_nhwc(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == 3:
        return jnp.transpose(x, (1, 2, 0))
    if x.ndim == 4:
        return jnp.transpose(x, (0, 2, 3, 1))
    return x


def _to_nchw(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == 3:
        return jnp.transpose(x, (2, 0, 1))
    if x.ndim == 4:
        return jnp.transpose(x, (0, 3, 1, 2))
    return x
class ConvResBlock(nn.Module):
    """Post-activation residual block from HyperCEZ ``base_model``."""

    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _to_nhwc(x)
        x = x[None, ...] if x.ndim == 3 else x
        residual = x
        y = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False)(x)
        y = nn.relu(y)
        y = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False)(y)
        x = nn.relu(residual + y)
        return _to_nchw(x[0] if x.ndim == 4 else x)


class DownSample(nn.Module):
    """HyperCEZ downsampling tower (approx. /16 spatial reduction)."""

    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half = self.out_channels // 2
        x = _to_nhwc(x)[None, ...]
        x = nn.Conv(half, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.relu(x)
        x = ConvResBlock(half)(_to_nchw(x[0]))
        x = _to_nhwc(x)[None, ...]
        x = nn.Conv(self.out_channels, (3, 3), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.relu(x)
        x = ConvResBlock(self.out_channels)(_to_nchw(x[0]))
        x = _to_nhwc(x)[None, ...]
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        x = ConvResBlock(self.out_channels)(_to_nchw(x[0]))
        x = _to_nhwc(x)[None, ...]
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        return _to_nchw(x[0])


class VectorRepresentationNetwork(nn.Module):
    """HyperCEZ ``AltRepresentationNetwork``."""

    obs_dim: int
    n_stack: int
    num_blocks: int
    rep_net_shape: int
    hidden_shape: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs.reshape(-1)
        expected = self.obs_dim * self.n_stack
        if x.shape[0] != expected:
            x = jnp.pad(x, (0, max(0, expected - x.shape[0])))[:expected]
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_shape)(x)
        x = nn.LayerNorm()(x)
        x = jnp.tanh(x)
        for _ in range(self.num_blocks):
            x = ImproveResidualBlock(self.hidden_shape, self.rep_net_shape)(x)
        return x


class VectorDynamicsNetwork(nn.Module):
    """HyperCEZ ``AltDynamicsNetwork``."""

    hidden_shape: int
    action_dim: int
    num_blocks: int
    dyn_shape: int
    act_embed_shape: int

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        action_vec = jnp.asarray(action, dtype=jnp.float32).reshape(-1)
        if action_vec.shape[0] != self.action_dim:
            action_vec = jnp.pad(action_vec, (0, max(0, self.action_dim - action_vec.shape[0])))[
                : self.action_dim
            ]
        act_emb = nn.Dense(self.act_embed_shape)(action_vec)
        act_emb = nn.LayerNorm()(act_emb)
        act_emb = nn.relu(act_emb)
        x = nn.LayerNorm()(jnp.concatenate([state, act_emb], axis=-1))
        x = nn.Dense(self.dyn_shape)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_shape)(x)
        next_state = state + x
        for _ in range(self.num_blocks):
            next_state = ImproveResidualBlock(self.hidden_shape, self.dyn_shape)(next_state)
        return next_state


class VectorValuePolicyNetwork(nn.Module):
    """HyperCEZ ``AltValuePolicyNetwork``."""

    hidden_shape: int
    val_net_shape: Sequence[int]
    pi_net_shape: Sequence[int]
    policy_output_size: int
    value_output_size: int
    v_num: int
    init_zero: bool = True
    policy_distribution: str = "squashed_gaussian"
    use_bn: bool = False

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        value_hidden = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)(state)
        value_hidden = nn.LayerNorm()(value_hidden)
        values = []
        for index in range(self.v_num):
            values.append(
                HyperMLP(
                    [*_layer_sizes(self.val_net_shape), self.value_output_size],
                    use_layernorm=self.use_bn,
                    zero_last=False,
                    name=f"value_head_{index}",
                )(value_hidden)
            )
        value = jnp.stack(values, axis=0)

        policy_hidden = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)(state)
        policy_hidden = nn.LayerNorm()(policy_hidden)
        policy = HyperMLP(
            [*_layer_sizes(self.pi_net_shape), self.policy_output_size],
            use_layernorm=False,
            zero_last=self.init_zero,
            name="policy_head",
        )(policy_hidden)

        if self.policy_distribution == "squashed_gaussian":
            action_dim = self.policy_output_size // 2
            mu = 5.0 * jnp.tanh(policy[:action_dim] / 5.0)
            std = jax.nn.softplus(policy[action_dim:] + 1.0) + 0.1
            std = jnp.clip(std, 0.1, 10.0)
            policy = jnp.concatenate([mu, std], axis=-1)

        return value, policy


class VectorRewardNetwork(nn.Module):
    """HyperCEZ ``AltRewardNetwork``."""

    hidden_shape: int
    rew_net_shape: Sequence[int]
    output_size: int
    init_zero: bool = True
    use_bn: bool = False

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)(state)
        x = nn.LayerNorm()(x)
        return HyperMLP(
            [*_layer_sizes(self.rew_net_shape), self.output_size],
            use_layernorm=self.use_bn,
            zero_last=self.init_zero,
        )(x)


class VectorRewardLSTMNetwork(nn.Module):
    """HyperCEZ ``AltRewardNetworkLSTM``."""

    hidden_shape: int
    rew_net_shape: Sequence[int]
    output_size: int
    lstm_hidden_size: int
    init_zero: bool = True
    use_bn: bool = False

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        reward_hidden: tuple[jnp.ndarray, jnp.ndarray] | None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        x = ImproveResidualBlock(self.hidden_shape, self.hidden_shape)(state)
        x = nn.LayerNorm()(x)
        x = x[None, ...]
        cell = nn.LSTMCell(self.lstm_hidden_size)
        carry = (
            reward_hidden
            if reward_hidden is not None
            else cell.initialize_carry(jax.random.PRNGKey(0), (1,))
        )
        carry, y = cell(carry, x[0])
        reward = HyperMLP(
            [*_layer_sizes(self.rew_net_shape), self.output_size],
            use_layernorm=self.use_bn,
            zero_last=self.init_zero,
        )(y)
        return reward, carry


class VectorProjectionNetwork(nn.Module):
    hidden_shape: int
    proj_hid_shape: int
    proj_shape: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.proj_hid_shape)(state)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.proj_hid_shape)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.proj_shape)(x)
        return nn.LayerNorm()(x)


class VectorProjectionHeadNetwork(nn.Module):
    proj_shape: int
    pred_hid_shape: int
    pred_shape: int

    @nn.compact
    def __call__(self, proj: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.pred_hid_shape)(proj)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.pred_shape)(x)


class ConvRepresentationNetwork(nn.Module):
    """HyperCEZ ``BaseRepresentationNetwork``."""

    input_shape: tuple[int, int, int]
    num_blocks: int
    num_channels: int
    down_sample: bool

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs.reshape(self.input_shape) if obs.ndim == 1 else obs
        if self.down_sample:
            x = DownSample(self.num_channels)(x)
        else:
            x = _to_nhwc(x)[None, ...]
            x = nn.Conv(self.num_channels, (3, 3), padding="SAME", use_bias=False)(x)
            x = nn.relu(_to_nchw(x[0]))
        for _ in range(self.num_blocks):
            x = ConvResBlock(self.num_channels)(x)
        return x


class ConvDynamicsNetwork(nn.Module):
    """HyperCEZ ``BaseDynamicsNetwork``."""

    num_blocks: int
    num_channels: int
    num_actions: int
    action_embedding: bool
    action_embedding_dim: int
    continuous: bool = False

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        _, height, width = state.shape
        if self.continuous:
            action_vec = jnp.asarray(action, dtype=jnp.float32).reshape(-1)
            action_plane = jnp.broadcast_to(action_vec.reshape(-1, 1, 1), (action_vec.shape[0], height, width))
        else:
            action_scalar = jnp.asarray(action, dtype=jnp.float32).reshape(())
            action_plane = jnp.full((1, height, width), action_scalar / self.num_actions)

        state_nhwc = _to_nhwc(state)[None, ...]
        if self.action_embedding:
            action_nhwc = _to_nhwc(action_plane)[None, ...]
            action_feat = nn.Conv(self.action_embedding_dim, (1, 1), padding="VALID", use_bias=False)(action_nhwc)
            action_feat = nn.LayerNorm(reduction_axes=(1, 2))(action_feat)
            action_feat = nn.relu(action_feat)
            x = jnp.concatenate([state_nhwc, action_feat], axis=-1)
            x = nn.Conv(self.num_channels, (3, 3), padding="SAME", use_bias=False)(x)
        else:
            action_nhwc = _to_nhwc(action_plane)[None, ...]
            x = jnp.concatenate([state_nhwc, action_nhwc], axis=-1)
            x = nn.Conv(self.num_channels, (3, 3), padding="SAME", use_bias=False)(x)

        x = nn.relu(x)
        x = x + state_nhwc
        x = nn.relu(x)
        x = _to_nchw(x[0])
        for _ in range(self.num_blocks):
            x = ConvResBlock(self.num_channels)(x)
        return x


class ConvValuePolicyNetwork(nn.Module):
    """HyperCEZ ``BaseValuePolicyNetwork``."""

    num_blocks: int
    num_channels: int
    reduced_channels: int
    flatten_size: int
    fc_layers: Sequence[int]
    value_output_size: int
    policy_output_size: int
    v_num: int
    init_zero: bool = True
    continuous: bool = False

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = state
        for _ in range(self.num_blocks):
            x = ConvResBlock(self.num_channels)(x)

        values = []
        for index in range(self.v_num):
            value = nn.Conv(
                self.reduced_channels,
                (1, 1),
                padding="VALID",
                use_bias=False,
                name=f"value_conv_{index}",
            )(_to_nhwc(x)[None, ...])
            value = nn.relu(value)
            value = value.reshape(-1)
            if value.shape[0] != self.flatten_size:
                value = jnp.pad(value, (0, max(0, self.flatten_size - value.shape[0])))[: self.flatten_size]
            value = HyperMLP(
                [*_layer_sizes(self.fc_layers), self.value_output_size],
                use_layernorm=False,
                zero_last=False if self.continuous else self.init_zero,
                name=f"value_fc_{index}",
            )(value)
            values.append(value)
        value = jnp.stack(values, axis=0)

        policy = nn.Conv(
            self.reduced_channels,
            (1, 1),
            padding="VALID",
            use_bias=False,
            name="policy_conv",
        )(_to_nhwc(x)[None, ...])
        policy = nn.relu(policy)
        policy = policy.reshape(-1)
        if policy.shape[0] != self.flatten_size:
            policy = jnp.pad(policy, (0, max(0, self.flatten_size - policy.shape[0])))[: self.flatten_size]
        policy_hidden = (
            HyperMLP([64], use_layernorm=False, zero_last=False, name="policy_fc_hidden")(policy)
            if self.continuous
            else HyperMLP([*_layer_sizes(self.fc_layers)], use_layernorm=False, zero_last=False, name="policy_fc_hidden")(
                policy
            )
        )
        policy = HyperMLP(
            [self.policy_output_size],
            use_layernorm=False,
            zero_last=self.init_zero,
            name="policy_fc_out",
        )(policy_hidden)

        if self.continuous:
            action_dim = self.policy_output_size // 2
            mu = 5.0 * jnp.tanh(policy[:action_dim] / 5.0)
            std = jax.nn.softplus(policy[action_dim:] + 1.0) + 0.1
            policy = jnp.concatenate([mu, std], axis=-1)

        return value, policy


class ConvSupportNetwork(nn.Module):
    num_channels: int
    reduced_channels: int
    flatten_size: int
    fc_layers: Sequence[int]
    output_size: int
    init_zero: bool = True

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(self.reduced_channels, (1, 1), padding="VALID", use_bias=False)(
            _to_nhwc(state)[None, ...]
        )
        x = nn.relu(x)
        x = x.reshape(-1)
        if x.shape[0] != self.flatten_size:
            x = jnp.pad(x, (0, max(0, self.flatten_size - x.shape[0])))[: self.flatten_size]
        return HyperMLP(
            [*_layer_sizes(self.fc_layers), self.output_size],
            use_layernorm=False,
            zero_last=self.init_zero,
        )(x)


class ConvSupportLSTMNetwork(nn.Module):
    num_channels: int
    reduced_channels: int
    flatten_size: int
    fc_layers: Sequence[int]
    output_size: int
    lstm_hidden_size: int
    init_zero: bool = True

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        reward_hidden: tuple[jnp.ndarray, jnp.ndarray] | None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        x = nn.Conv(self.reduced_channels, (1, 1), padding="VALID", use_bias=False)(
            _to_nhwc(state)[None, ...]
        )
        x = nn.relu(x)
        x = x.reshape(-1)
        if x.shape[0] != self.flatten_size:
            x = jnp.pad(x, (0, max(0, self.flatten_size - x.shape[0])))[: self.flatten_size]
        x = x[None, ...]
        cell = nn.LSTMCell(self.lstm_hidden_size)
        carry = (
            reward_hidden
            if reward_hidden is not None
            else cell.initialize_carry(jax.random.PRNGKey(0), (1,))
        )
        carry, y = cell(carry, x[0])
        y = nn.LayerNorm()(y)
        y = nn.relu(y)
        prefix = HyperMLP(
            [*_layer_sizes(self.fc_layers), self.output_size],
            use_layernorm=False,
            zero_last=self.init_zero,
        )(y)
        return prefix, carry


class DenseProjectionNetwork(nn.Module):
    """HyperCEZ ``BaseProjectionNetwork`` (image models)."""

    state_dim: int
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = state.reshape(-1)
        if x.shape[0] != self.state_dim:
            x = jnp.pad(x, (0, max(0, self.state_dim - x.shape[0])))[: self.state_dim]
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return nn.LayerNorm()(x)


class DenseProjectionHeadNetwork(nn.Module):
    """HyperCEZ ``BaseProjectionHeadNetwork``."""

    in_dim: int
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, proj: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(proj)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)


def conv_state_shape(
    obs_shape: tuple[int, int, int],
    *,
    num_channels: int,
    down_sample: bool,
) -> tuple[int, int, int]:
    height, width = obs_shape[1], obs_shape[2]
    if down_sample:
        height = math.ceil(height / 16)
        width = math.ceil(width / 16)
    return (num_channels, height, width)


def flatten_spatial_shape(
    state_shape: tuple[int, int, int],
    *,
    reduced_channels: int,
) -> tuple[int, int]:
    channels, height, width = state_shape
    state_dim = channels * height * width
    flatten_size = reduced_channels * height * width
    return state_dim, flatten_size
