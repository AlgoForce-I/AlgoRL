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
    checkpoint_freq: int | None = None
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
    reanalyze_ratio: float = 0.5
    unroll_steps: int = 5


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
