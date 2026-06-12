"""EfficientZero agent."""

from __future__ import annotations

from algorl.agents._composed import ComposedAgent
from algorl.agents.configs import EfficientZeroConfig


class EfficientZero(ComposedAgent):
    """EfficientZero search-based agent."""

    composition_name = "efficient_zero"
    config_class = EfficientZeroConfig
