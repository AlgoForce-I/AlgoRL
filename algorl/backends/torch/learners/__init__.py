"""PyTorch learner registry (stub).

Register torch learners here once the PyTorch backend is implemented.
"""

from __future__ import annotations

from algorl.core.learner import Learner
from algorl.core.registry import KindRegistry

registry: KindRegistry[Learner] = KindRegistry("PyTorch learner")

# Register torch implementations with @registry.register("kind") when ready.
