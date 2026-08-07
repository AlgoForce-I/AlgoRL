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
    burst_compile_steps: int | None = None
    checkpoint_freq: int | None = None
    # Directory for multi-file run checkpoints (periodic, boundary, best/).
    checkpoint_dir: str | None = None
    checkpoint_at_task_boundary: bool = True
    checkpoint_keep_last: int | None = None
    # When True, keep ``{checkpoint_dir}/best/`` for the highest return so far.
    autosave_best: bool = False
    autosave_best_metric: str = "mean_episode_return"
    autosave_best_window: int = 10
    autosave_best_min_step: int = 0
    jax_rollout_chunk: int = 64
    # Log host RAM + device VRAM scalars to TensorBoard every N env steps (0=off).
    # Also forced once after each post-rollout gradient burst (captures train peaks).
    memory_log_interval: int = 1_000
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

    @property
    def rollout_envs(self) -> int:
        """Parallel rollout lanes configured for MCTS (``search_batch_size``)."""
        return max(1, self.search_batch_size)

    @property
    def uses_batched_rollout(self) -> bool:
        """Whether rollout MCTS plans across multiple env lanes at once."""
        return self.search_batch_size > 1


@dataclass(frozen=True)
class EfficientZeroConfig(SearchAgentConfig):
    """EfficientZero-V2 hyperparameters.

    Use :meth:`for_sequential` or :meth:`for_batched` for the common training
    layouts. Lower-level presets (:meth:`for_dmc_state`, :meth:`for_atari`, …)
    target specific observation modalities.
    """

    reanalyze_ratio: float = 0.5
    # Reanalyze MCTS width: an int, ``None`` (fall back to ``search_batch_size``),
    # or ``"auto"`` (size from free GPU memory at learner init; results are
    # identical at any width, wider just raises device occupancy).
    reanalyze_search_batch_size: int | str | None = None
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
    # When True, copy learner weights into self-play MCTS before each batched
    # rollout chunk. Default False matches EZ-V2 (refresh only every
    # ``self_play_update_interval`` train steps).
    sync_self_play_before_rollout: bool = False
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
        batches ``reanalyze_search_batch_size`` roots per JIT search
        (``"auto"`` sizes the width from free GPU memory).
        """
        config = cls.for_dmc_state(
            search_batch_size=1,
            reanalyze_search_batch_size="auto",
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

        Learner and reanalyze settings match :meth:`for_sequential`. Rollout
        parallelism uses ``search_batch_size=num_envs`` so each MCTS call plans
        for all lanes at once; ``jax_rollout_chunk`` stays at the sequential
        value so a chunk still runs multiple batched MCTS steps before training.

        Training runs as a post-rollout burst of ``num_envs * jax_rollout_chunk``
        gradient steps (one update per collected env step, matching sequential).
        The burst executes in sub-bursts of ``burst_compile_steps`` updates:
        each sub-burst re-samples the buffer, recomputes value targets, and
        reruns fused reanalyze against the latest parameters and priorities,
        keeping target staleness close to the sequential loop while the scanned
        optimizer and wide reanalyze searches keep GPU throughput high. Set
        ``gradient_steps_per_rollout=None`` for fully interleaved training.
        """
        dynamics_every = 10
        if overrides and "dynamics_update_every" in overrides:
            dynamics_every = int(overrides["dynamics_update_every"])  # type: ignore[arg-type]
        burst_steps = num_envs * dynamics_every
        config = cls.for_sequential(
            search_batch_size=num_envs,
            jax_rollout_chunk=dynamics_every,
            dynamics_update_every=dynamics_every,
            gradient_steps_per_rollout=burst_steps,
            # Sub-burst width: bounded so priorities/targets refresh at least as
            # often as the sequential self-play interval, while each sub-burst
            # still feeds reanalyze enough roots to saturate the search width.
            burst_compile_steps=min(burst_steps, 64),
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


DEFAULT_HYPERCEZ_HNET_COMPONENTS: tuple[str, ...] = (
    "representation_model",
    "dynamics_model",
    "reward_prediction_model",
    "value_policy_model",
)


@dataclass(frozen=True)
class HyperCEZConfig(EfficientZeroConfig):
    """HyperCEZDelta on top of EfficientZero.

    Defaults match the CW10 continual-learning recipe used by
    ``example_hypercez.py``. Task-conditioned hypernetworks emit weight
    deltas for selected EZ components; LayerNorm / obs-norm stats and
    projection nets stay shared.

    ``hnet_type`` selects unchunked (one head per weight tensor; default) or
    chunked HyperCL-style generators (``chunk_dim`` / ``cemb_size``).
    """

    # EZ training knobs (re-applied in presets: ``for_dmc_state`` overrides some).
    use_bn: bool = True
    lr_warm_up: float = 0.01
    clip_inference_values: bool = True
    change_temperature: bool = False
    reward_support_range: tuple[float, float] = (-10.0, 10.0)
    discount: float = 0.99
    value_support_range: tuple[float, float] = (-1000.0, 1000.0)
    mcts_simulations: int = 64
    max_num_considered_actions: int | None = 16
    entropy_coeff: float = 0.1
    std_magnification: float = 4.0
    schedule_horizon: str = "fixed"
    lr_decay_steps: int = 300_000
    lr_decay_rate: float = 0.5

    hnet_components: tuple[str, ...] = DEFAULT_HYPERCEZ_HNET_COMPONENTS
    hnet_arch: tuple[int, ...] = (128, 128)
    hnet_type: str = "unchunked"
    chunk_dim: int = 2000
    cemb_size: int = 20
    cemb_init_std: float = 1.0
    emb_size: int = 10
    num_tasks: int = 10
    lr_hyper: float = 3e-4
    beta: float = 0.5
    # Unit-scale residual cap so ΔW stays on the same order as EZ weight updates
    # (α_max=0.2 forced hypernet outputs ~5× larger for the same ΔW_eff).
    alpha_max: float = 1.0
    # Open residual path; unchunked heads scale by 1/sqrt(H) at init.
    alpha_init: float = 2.0
    emb_init_std: float = 1.0
    # Identity residual at step 0; pair with unchunked 1/sqrt(H) head scale.
    head_init_std: float = 0.0
    no_look_ahead: bool = False
    dt_scale: float = 1.0
    use_sgd_change: bool = False
    plastic_prev_tembs: bool = False
    ewc_weight_importance: bool = False
    hnet_grad_max_norm: float = 5.0
    steps_per_task: int | None = 1_000_000  # per-task schedule / LR horizon
    scale_hyper_lr: bool = False  # False: hypernet/α keep full lr_hyper
    # True: W0 (generated base) stays frozen. False: optimize W0 slowly with
    # lr_W0 = lr_hyper / lr_main_to_lr_hyper_ratio (task-shared backbone).
    frozen_base_weights: bool = True
    lr_main_to_lr_hyper_ratio: float = 50.0
    warm_start_alpha: bool = True  # α_t ← α_{t-1} at task boundary
    snapshot_shared_per_task: bool = True  # snapshot LN / obs-norm per task
    use_per_task_reg_scaling: bool = False  # off: dynamic β is enough; inv-EMA fights retention
    reg_scaling_min: float = 0.25
    reg_scaling_max: float = 4.0
    retention_log_interval: int = 500  # log fix-target drift; 0 disables

    @classmethod
    def _cw_training_overrides(cls) -> dict[str, object]:
        """CW10-style EZ knobs (parent presets may overwrite dataclass defaults)."""
        return {
            "use_bn": True,
            "lr_warm_up": 0.01,
            "clip_inference_values": True,
            "change_temperature": False,
            "reward_support_range": (-10.0, 10.0),
            "discount": 0.99,
            "value_support_range": (-1000.0, 1000.0),
            "mcts_simulations": 64,
            "max_num_considered_actions": 16,
            "entropy_coeff": 0.1,
            "std_magnification": 4.0,
            "schedule_horizon": "fixed",
            "lr_decay_steps": 300_000,
            "lr_decay_rate": 0.5,
        }

    @classmethod
    def for_sequential(cls, **overrides: object) -> HyperCEZConfig:
        """Single-env HyperCEZ with CW10 training defaults."""
        config = super().for_sequential().with_overrides(**cls._cw_training_overrides())
        return config.with_overrides(**overrides) if overrides else config

    @classmethod
    def for_batched(
        cls,
        *,
        num_envs: int,
        **overrides: object,
    ) -> HyperCEZConfig:
        """Batched HyperCEZ with CW10 defaults and a per-task mix/TD schedule.

        Schedule horizons are derived from ``steps_per_task`` (default 1M),
        matching task-0 EfficientZero rather than the full multi-task run.
        ``lr_decay_*`` stay fixed afterward.
        """
        steps_per_task = int(overrides.get("steps_per_task", 1_000_000))
        cw = {
            key: value
            for key, value in cls._cw_training_overrides().items()
            if key not in {"schedule_horizon", "lr_decay_steps", "lr_decay_rate"}
        }
        config = (
            super()
            .for_batched(num_envs=num_envs)
            .with_overrides(
                **cw,
                # Resolve mix/TD against the per-task horizon while horizon is auto.
                schedule_horizon="auto",
                steps_per_task=steps_per_task,
            )
            .with_schedule_for_run(steps_per_task, num_envs=num_envs)
            .with_overrides(
                schedule_horizon="fixed",
                lr_decay_steps=300_000,
                lr_decay_rate=0.5,
            )
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
