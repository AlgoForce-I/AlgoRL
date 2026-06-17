"""Random seed utilities."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs."""
    random.seed(seed)
    np.random.seed(seed)
