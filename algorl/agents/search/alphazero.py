"""AlphaZero agent."""

from __future__ import annotations

from algorl.agents._composed import ComposedAgent
from algorl.agents.configs import AlphaZeroConfig


class AlphaZero(ComposedAgent):
    """AlphaZero search-based agent."""

    composition_name = "alphazero"
    config_class = AlphaZeroConfig
