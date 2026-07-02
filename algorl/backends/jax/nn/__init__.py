"""Flax network modules for the JAX backend.

Each algorithm family lives in its own subpackage under ``nn/``.
"""

from algorl.backends.jax.nn.efficientzero import (
    EfficientZero,
    build_efficient_zero_model,
    build_efficient_zero_model_from_env,
    init_efficient_zero_params,
    init_efficient_zero_params_from_env,
    init_efficient_zero_params_from_model,
    infer_model_type,
)

__all__ = [
    "EfficientZero",
    "build_efficient_zero_model",
    "build_efficient_zero_model_from_env",
    "init_efficient_zero_params",
    "init_efficient_zero_params_from_env",
    "init_efficient_zero_params_from_model",
    "infer_model_type",
]
