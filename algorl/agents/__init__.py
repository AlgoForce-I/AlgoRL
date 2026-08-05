"""User-facing agents and config classes.

Agent classes are imported lazily so ``algorl.agents.configs`` can be loaded
without pulling in planners/learners (avoids circular imports via ``envs.resolve``).
"""

from __future__ import annotations

import importlib
from typing import Any

from algorl.agents.configs import (
    AlphaZeroConfig,
    BaseAgentConfig,
    DreamerV3Config,
    EfficientZeroConfig,
    HyperCEZConfig,
    MuZeroConfig,
    PlaNetConfig,
    SearchAgentConfig,
    TDMPCConfig,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AlphaZero": ("algorl.agents.search.alphazero", "AlphaZero"),
    "DreamerV3": ("algorl.agents.world_model.dreamer_v3", "DreamerV3"),
    "EfficientZero": ("algorl.agents.search.efficient_zero", "EfficientZero"),
    "HyperCEZ": ("algorl.agents.search.hyper_cez", "HyperCEZ"),
    "MuZero": ("algorl.agents.search.muzero", "MuZero"),
    "PlaNet": ("algorl.agents.world_model.planet", "PlaNet"),
    "TDMPC": ("algorl.agents.world_model.td_mpc", "TDMPC"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    return getattr(importlib.import_module(module_name), attr_name)


__all__ = [
    "AlphaZero",
    "AlphaZeroConfig",
    "BaseAgentConfig",
    "DreamerV3",
    "DreamerV3Config",
    "EfficientZero",
    "EfficientZeroConfig",
    "HyperCEZ",
    "HyperCEZConfig",
    "MuZero",
    "MuZeroConfig",
    "PlaNet",
    "PlaNetConfig",
    "SearchAgentConfig",
    "TDMPC",
    "TDMPCConfig",
]
