"""Shared backend-agnostic data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

ArrayLike = Any
Observation = Any
Action = Any
Reward = float
Done = bool
Info = Mapping[str, Any]


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
    """A batch of training data."""

    data: Mapping[str, ArrayLike]
