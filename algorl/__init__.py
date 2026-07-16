"""AlgoRL: unified model-based reinforcement learning."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
from typing import Any

# Gymnasium AsyncVectorEnv (spawn) re-imports user scripts in worker processes.
# Pin workers to CPU JAX before any optional backend import can claim GPU memory.
if mp.current_process().name != "MainProcess":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AlphaZero": ("algorl.agents.search.alphazero", "AlphaZero"),
    "AlphaZeroConfig": ("algorl.agents.configs", "AlphaZeroConfig"),
    "BaseAgentConfig": ("algorl.agents.configs", "BaseAgentConfig"),
    "DreamerV3": ("algorl.agents.world_model.dreamer_v3", "DreamerV3"),
    "DreamerV3Config": ("algorl.agents.configs", "DreamerV3Config"),
    "EfficientZero": ("algorl.agents.search.efficient_zero", "EfficientZero"),
    "EfficientZeroConfig": ("algorl.agents.configs", "EfficientZeroConfig"),
    "MuZero": ("algorl.agents.search.muzero", "MuZero"),
    "MuZeroConfig": ("algorl.agents.configs", "MuZeroConfig"),
    "PlaNet": ("algorl.agents.world_model.planet", "PlaNet"),
    "PlaNetConfig": ("algorl.agents.configs", "PlaNetConfig"),
    "SearchAgentConfig": ("algorl.agents.configs", "SearchAgentConfig"),
    "TDMPC": ("algorl.agents.world_model.td_mpc", "TDMPC"),
    "TDMPCConfig": ("algorl.agents.configs", "TDMPCConfig"),
    "get_backend": ("algorl.core.factory", "get_backend"),
    "make_env": ("algorl.envs", "make_env"),
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
    "MuZero",
    "MuZeroConfig",
    "PlaNet",
    "PlaNetConfig",
    "SearchAgentConfig",
    "TDMPC",
    "TDMPCConfig",
    "get_backend",
    "make_env",
    "__version__",
]

__version__ = "0.0.1"
