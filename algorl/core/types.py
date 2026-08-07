"""Shared backend-agnostic data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import jax.numpy as jnp

Observation: TypeAlias = npt.NDArray[Any] | Mapping[str, npt.NDArray[Any]] | int | float
Action: TypeAlias = int | npt.NDArray[np.floating[Any]] | jnp.ndarray
Reward: TypeAlias = float
Done: TypeAlias = bool
Info: TypeAlias = Mapping[str, Any]

# Latent states and backend tensors stay opaque at the public API boundary.
LatentState: TypeAlias = Any
ArrayLike: TypeAlias = npt.NDArray[Any] | list[float] | tuple[float, ...]
PolicyTarget: TypeAlias = npt.NDArray[np.floating[Any]] | jnp.ndarray
ValueTarget: TypeAlias = float | npt.NDArray[np.floating[Any]]

T = TypeVar("T")


@dataclass
class Transition:
    """A single environment transition."""

    observation: Observation
    action: Action
    reward: Reward
    next_observation: Observation
    done: Done
    info: Info = field(default_factory=dict)


@dataclass
class Batch:
    """A batch of training data keyed by field name."""

    data: Mapping[str, ArrayLike | list[Transition] | list[Any]]


@dataclass
class SearchEntry:
    """One MCTS search outcome stored for training or reanalyze."""

    observation: Observation
    policy_target: PolicyTarget
    value_target: ValueTarget
    model_version: int = 0
