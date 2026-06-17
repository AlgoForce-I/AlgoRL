"""TD-MPC agent."""

from __future__ import annotations

from algorl.agents._composed import ComposedAgent
from algorl.agents.configs import TDMPCConfig


class TDMPC(ComposedAgent):
    """TD-MPC / TD-MPC2 world-model agent."""

    composition_name = "td_mpc"
    config_class = TDMPCConfig
