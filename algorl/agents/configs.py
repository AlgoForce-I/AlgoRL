"""Typed hyperparameter configs for AlgoRL agents."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class BaseAgentConfig:
    """Common training and composition settings."""

    backend: str = "jax"
    seed: int = 0
    buffer_capacity: int = 100_000
    batch_size: int = 32
    train_freq: int = 1
    learning_starts: int = 1_000
    gradient_steps_per_rollout: int | None = None
    checkpoint_freq: int | None = None
    jax_rollout_chunk: int = 64
    require_implemented: bool = True

    def with_overrides(self, **overrides: object) -> BaseAgentConfig:
        """Return a copy with only known config fields replaced."""
        valid = {field.name for field in fields(self)}
        unknown = set(overrides) - valid
        if unknown:
            unknown_fields = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown config field(s): {unknown_fields}")
        return replace(self, **overrides)


@dataclass(frozen=True)
class SearchAgentConfig(BaseAgentConfig):
    """Shared MCTS settings for search-based agents."""

    mcts_simulations: int = 50
    search_batch_size: int = 1
    mcts_temperature: float = 1.0
    dirichlet_fraction: float = 0.0
    dirichlet_alpha: float = 0.3
    gumbel_scale: float = 1.0
    max_num_considered_actions: int | None = None


@dataclass(frozen=True)
class EfficientZeroConfig(SearchAgentConfig):
    """EfficientZero-V2 hyperparameters.

    Use :meth:`for_sequential` or :meth:`for_batched` for the common training
    layouts. Lower-level presets (:meth:`for_dmc_state`, :meth:`for_atari`, …)
    target specific observation modalities.
    """

    reanalyze_ratio: float = 0.5
    reanalyze_search_batch_size: int | None = None
    reanalyze_mini_batch_size: int = 256
    reanalyze_update_interval: int = 200
    unroll_steps: int = 5
    trajectory_size: int = 100
    learning_rate: float = 3e-4
    lr_warm_up: float = 0.005
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 300_000
    weight_decay: float = 0.0
    reward_loss_coeff: float = 1.0
    value_loss_coeff: float = 0.5
    policy_loss_coeff: float = 1.0
    consistency_coeff: float = 2.0
    entropy_coeff: float = 0.05
    max_grad_norm: float = 5.0
    use_IQL: bool = False
    IQL_weight: float = 0.5
    value_support_range: tuple[float, float] = (-299.0, 299.0)
    reward_support_range: tuple[float, float] = (-2.0, 2.0)
    discount: float = 0.997
    td_steps: int = 5
    td_lambda: float = 0.95
    gae_max_steps: int = 15
    lstm_horizon_len: int = 5
    value_target: str = "mixed"
    model_value_target: str = "bootstrapped"
    start_use_mix_training_steps: int = 40_000
    mixed_value_threshold: float = 20_000.0
    auto_td_steps: int = 30_000
    self_play_update_interval: int = 100
    change_temperature: bool = True
    total_training_steps: int = 100_000
    dynamics_update_every: int = 10
    std_magnification: float = 3.0
    use_priority: bool = True
    priority_prob_alpha: float = 1.0
    priority_prob_beta: float = 1.0
    min_prior: float = 1e-6
    top_transitions: float = 200_000.0
    policy_action_num: int = 4
    random_action_num: int = 12
    state_norm: bool = False
    value_prefix: bool = False
    v_num: int = 1
    value_support_type: str = "support"
    reward_support_type: str = "support"
    support_bins: int = 51
    clip_inference_values: bool = True
    model_type: str = "auto"
    n_stack: int = 1
    num_blocks: int = 2
    num_channels: int = 64
    reduced_channels: int = 16
    fc_layers: tuple[int, ...] = (32,)
    down_sample: bool = True
    init_zero: bool = True
    action_embedding: bool = True
    action_embedding_dim: int = 16
    lstm_hidden_size: int = 512
    projection_layers: tuple[int, int] = (1024, 1024)
    projection_head_layers: tuple[int, int] = (256, 1024)
    hidden_shape: int = 128
    rep_net_shape: int = 256
    dyn_shape: int = 256
    act_embed_shape: int = 64
    val_net_shape: tuple[int, ...] = (256, 256)
    pi_net_shape: tuple[int, ...] = (256, 256)
    rew_net_shape: tuple[int, ...] = (256, 256)
    proj_hid_shape: int = 512
    proj_shape: int = 128
    pred_hid_shape: int = 512
    pred_shape: int = 128
    policy_distribution: str = "squashed_gaussian"
    use_gumbel: bool = False
    value_policy_detach: bool = False
    use_bn: bool = False
    use_p_norm: bool = False
    noisy_net: bool = False
    schedule_horizon: str = "auto"
    schedule_mix_start_fraction: float | None = None
    schedule_auto_td_fraction: float | None = None
    schedule_mixed_value_buffer_fraction: float | None = None

    def with_schedule_for_run(
        self,
        total_timesteps: int,
        *,
        num_envs: int = 1,
    ) -> EfficientZeroConfig:
        """Apply :func:`~algorl.buffers.efficientzero.schedule.resolve_efficient_zero_schedule`."""
        from algorl.buffers.efficientzero.schedule import resolve_efficient_zero_schedule

        return resolve_efficient_zero_schedule(
            self,
            total_timesteps,
            num_envs=num_envs,
        )

    @classmethod
    def for_sequential(cls, **overrides: object) -> EfficientZeroConfig:
        """Single-env vector control (Gymnasium / DMC state).

        Rollout MCTS uses ``search_batch_size=1``; training-time reanalyze
        batches ``reanalyze_search_batch_size`` roots per JIT search.
        """
        config = cls.for_dmc_state(
            search_batch_size=1,
            reanalyze_search_batch_size=2048,
            jax_rollout_chunk=10,
            gradient_steps_per_rollout=1,
            batch_size=256,
            reanalyze_ratio=1.0,
            seed=42,
            learning_starts=2_000,
        )
        return config.with_overrides(**overrides) if overrides else config

    @classmethod
    def for_batched(
        cls,
        *,
        num_envs: int,
        **overrides: object,
    ) -> EfficientZeroConfig:
        """Parallel vector-control envs with wide rollout MCTS.

        ``jax_rollout_chunk`` is chosen so ``chunk * num_envs`` matches
        ``dynamics_update_every`` (one learner update per rollout chunk).
        """
        dynamics_every = 10
        if overrides and "dynamics_update_every" in overrides:
            dynamics_every = int(overrides["dynamics_update_every"])  # type: ignore[arg-type]
        rollout_chunk = max(1, dynamics_every // max(1, num_envs))
        config = cls.for_dmc_state(
            batch_size=256,
            mcts_simulations=32,
            jax_rollout_chunk=rollout_chunk,
            search_batch_size=num_envs,
            reanalyze_ratio=1.0,
            gradient_steps_per_rollout=1,
            dynamics_update_every=dynamics_every,
            seed=0,
            buffer_capacity=10_000,
            learning_starts=1_000,
        )
        return config.with_overrides(**overrides) if overrides else config

    @classmethod
    def for_atari(cls, **overrides: object) -> EfficientZeroConfig:
        """EfficientZero-V2 Atari preset (discrete actions, image observations)."""
        config = cls(
            model_type="atari",
            value_prefix=True,
            n_stack=4,
            num_blocks=1,
            reduced_channels=16,
            action_embedding_dim=16,
            value_support_type="support",
            reward_support_type="support",
            support_bins=51,
            projection_layers=(1024, 1024),
            projection_head_layers=(256, 1024),
            policy_distribution="discrete",
            use_gumbel=True,
            reanalyze_ratio=1.0,
            weight_decay=1e-4,
        )
        return config.with_overrides(**overrides) if overrides else config

    @classmethod
    def for_dmc_image(cls, **overrides: object) -> EfficientZeroConfig:
        """EfficientZero-V2 DMC image preset."""
        config = cls(
            model_type="dmc_image",
            value_prefix=False,
            n_stack=4,
            num_blocks=1,
            reduced_channels=16,
            action_embedding_dim=16,
            value_support_type="support",
            reward_support_type="support",
            support_bins=51,
            projection_layers=(1024, 1024),
            projection_head_layers=(256, 1024),
            policy_distribution="squashed_gaussian",
            reanalyze_ratio=1.0,
            weight_decay=1e-4,
        )
        return config.with_overrides(**overrides) if overrides else config

    @classmethod
    def for_dmc_state(cls, **overrides: object) -> EfficientZeroConfig:
        """EfficientZero-V2 vector-control base preset."""
        config = cls(
            model_type="dmc_state",
            value_prefix=False,
            n_stack=1,
            num_blocks=2,
            hidden_shape=128,
            rep_net_shape=256,
            dyn_shape=256,
            act_embed_shape=64,
            rew_net_shape=(256, 256),
            val_net_shape=(256, 256),
            pi_net_shape=(256, 256),
            proj_hid_shape=512,
            proj_shape=128,
            pred_hid_shape=512,
            pred_shape=128,
            value_support_type="support",
            reward_support_type="support",
            support_bins=51,
            policy_distribution="squashed_gaussian",
            use_bn=False,
            use_p_norm=False,
            noisy_net=False,
            reanalyze_ratio=1.0,
            discount=0.997,
            td_steps=5,
            td_lambda=0.95,
            gae_max_steps=15,
            value_target="mixed",
            model_value_target="bootstrapped",
            start_use_mix_training_steps=40_000,
            auto_td_steps=60_000,
            self_play_update_interval=100,
            change_temperature=True,
            total_training_steps=100_000,
            dynamics_update_every=10,
            mixed_value_threshold=20_000.0,
            use_priority=True,
            entropy_coeff=0.05,
            consistency_coeff=2.0,
            policy_action_num=4,
            random_action_num=12,
            buffer_capacity=100_000,
            mcts_simulations=32,
            weight_decay=2e-5,
        )
        return config.with_overrides(**overrides) if overrides else config

    @classmethod
    def for_dmc_state_batched_cl(
        cls,
        *,
        num_envs: int,
        **overrides: object,
    ) -> EfficientZeroConfig:
        """Alias for :meth:`for_batched`."""
        return cls.for_batched(num_envs=num_envs, **overrides)

    @classmethod
    def for_dmc_state_batched_cl_gpu(
        cls,
        *,
        num_envs: int,
        **overrides: object,
    ) -> EfficientZeroConfig:
        """Alias for :meth:`for_batched`."""
        return cls.for_batched(num_envs=num_envs, **overrides)

    @classmethod
    def for_dmc_state_sequential_gpu(
        cls,
        **overrides: object,
    ) -> EfficientZeroConfig:
        """Alias for :meth:`for_sequential`."""
        return cls.for_sequential(**overrides)

    @classmethod
    def for_dmc_state_throughput(cls, **overrides: object) -> EfficientZeroConfig:
        """Vector-control preset tuned for maximum JAX throughput (no reanalyze)."""
        config = cls.for_dmc_state(
            reanalyze_ratio=0.0,
            batch_size=256,
            mcts_simulations=32,
            jax_rollout_chunk=256,
            search_batch_size=8,
        )
        return config.with_overrides(**overrides) if overrides else config


@dataclass(frozen=True)
class MuZeroConfig(SearchAgentConfig):
    unroll_steps: int = 5


@dataclass(frozen=True)
class AlphaZeroConfig(SearchAgentConfig):
    mcts_simulations: int = 100
    self_play_games: int = 1


@dataclass(frozen=True)
class DreamerV3Config(BaseAgentConfig):
    imagination_horizon: int = 15
    batch_length: int = 64


@dataclass(frozen=True)
class PlaNetConfig(BaseAgentConfig):
    cem_candidates: int = 1000
    cem_iterations: int = 10


@dataclass(frozen=True)
class TDMPCConfig(BaseAgentConfig):
    mpc_horizon: int = 12
    mpc_candidates: int = 64
