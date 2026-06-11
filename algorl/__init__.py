"""AlgoRL: unified model-based reinforcement learning."""

from algorl.agents.search.alphazero import AlphaZero
from algorl.agents.search.efficient_zero import EfficientZero
from algorl.agents.search.muzero import MuZero
from algorl.agents.world_model.dreamer_v3 import DreamerV3
from algorl.agents.world_model.planet import PlaNet
from algorl.agents.world_model.td_mpc import TDMPC
from algorl.core.factory import get_backend

__all__ = [
    "AlphaZero",
    "DreamerV3",
    "EfficientZero",
    "MuZero",
    "PlaNet",
    "TDMPC",
    "get_backend",
    "__version__",
]

__version__ = "0.0.1"
