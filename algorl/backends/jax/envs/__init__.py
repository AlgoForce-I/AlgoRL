"""JAX environment adapters.

Heavy optional backends (pgx, MTCWorld) are lazy-loaded so lightweight imports
such as :func:`make_gymnasium_vector_env` stay safe for Gymnasium spawn workers.
"""

from __future__ import annotations

import importlib
from typing import Any

from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.backends.jax.envs.gymnasium_env import GymnasiumSearchEnvironment, GymnasiumSearchEnv
from algorl.backends.jax.envs.gymnasium_vector import GymnasiumVectorJaxEnv, make_gymnasium_vector_env
from algorl.envs.jax_env import PolicyFn

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CW_BENCHMARK_NAMES": ("algorl.backends.jax.envs.mtcworld", "CW_BENCHMARK_NAMES"),
    "BatchedContinualLearningJaxEnv": (
        "algorl.backends.jax.envs.mtcworld_jax",
        "BatchedContinualLearningJaxEnv",
    ),
    "ContinualLearningJaxEnv": (
        "algorl.backends.jax.envs.mtcworld_jax",
        "ContinualLearningJaxEnv",
    ),
    "CWTaskJaxEnv": ("algorl.backends.jax.envs.mtcworld_jax", "CWTaskJaxEnv"),
    "MtcworldContinualGymEnv": ("algorl.backends.jax.envs.mtcworld", "MtcworldContinualGymEnv"),
    "MtcworldContinualRolloutCollector": (
        "algorl.backends.jax.envs.mtcworld_jax",
        "MtcworldContinualRolloutCollector",
    ),
    "MtcworldCWEvalGymEnv": ("algorl.backends.jax.envs.mtcworld", "MtcworldCWEvalGymEnv"),
    "MtcworldCWRolloutCollector": (
        "algorl.backends.jax.envs.mtcworld_jax",
        "MtcworldCWRolloutCollector",
    ),
    "MtcworldGymEnv": ("algorl.backends.jax.envs.mtcworld", "MtcworldGymEnv"),
    "MtcworldRolloutBatch": ("algorl.backends.jax.envs.mtcworld_jax", "MtcworldRolloutBatch"),
    "MtcworldRolloutCollector": (
        "algorl.backends.jax.envs.mtcworld_jax",
        "MtcworldRolloutCollector",
    ),
    "MtcworldCWSearchEnvironment": (
        "algorl.backends.jax.envs.mtcworld_search",
        "MtcworldCWSearchEnvironment",
    ),
    "MtcworldCWTaskSearchEnvironment": (
        "algorl.backends.jax.envs.mtcworld_search",
        "MtcworldCWTaskSearchEnvironment",
    ),
    "MtcworldSearchEnvironment": (
        "algorl.backends.jax.envs.mtcworld_search",
        "MtcworldSearchEnvironment",
    ),
    "PgxGymEnv": ("algorl.backends.jax.envs.pgx", "PgxGymEnv"),
    "PgxSearchEnvironment": ("algorl.backends.jax.envs.pgx", "PgxSearchEnvironment"),
    "SawyerJaxEnv": ("algorl.backends.jax.envs.mtcworld_jax", "SawyerJaxEnv"),
    "VectorJaxEnv": ("algorl.backends.jax.envs.mtcworld_jax", "VectorJaxEnv"),
    "as_mtcworld_jax_env": ("algorl.backends.jax.envs.mtcworld_jax", "as_mtcworld_jax_env"),
    "as_mtcworld_jax_env_from_spec": (
        "algorl.backends.jax.envs.mtcworld_jax",
        "as_mtcworld_jax_env_from_spec",
    ),
    "make_batched_cw_train_env": (
        "algorl.backends.jax.envs.mtcworld_jax",
        "make_batched_cw_train_env",
    ),
    "require_mtcworld": ("algorl.backends.jax.envs.mtcworld_jax", "require_mtcworld"),
    "search_env_from_context": ("algorl.backends.jax.envs.factory", "search_env_from_context"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    return getattr(importlib.import_module(module_name), attr_name)


__all__ = [
    "CW_BENCHMARK_NAMES",
    "BatchedContinualLearningJaxEnv",
    "ContinualLearningJaxEnv",
    "CWTaskJaxEnv",
    "GymnasiumSearchEnv",
    "GymnasiumSearchEnvironment",
    "GymnasiumVectorJaxEnv",
    "make_gymnasium_vector_env",
    "MtcworldContinualGymEnv",
    "MtcworldContinualRolloutCollector",
    "MtcworldCWEvalGymEnv",
    "MtcworldCWRolloutCollector",
    "MtcworldCWSearchEnvironment",
    "MtcworldCWTaskSearchEnvironment",
    "MtcworldGymEnv",
    "MtcworldRolloutBatch",
    "MtcworldRolloutCollector",
    "MtcworldSearchEnvironment",
    "PgxGymEnv",
    "PgxSearchEnvironment",
    "PolicyFn",
    "SawyerJaxEnv",
    "SearchEnvironment",
    "VectorJaxEnv",
    "as_mtcworld_jax_env",
    "as_mtcworld_jax_env_from_spec",
    "make_batched_cw_train_env",
    "require_mtcworld",
    "search_env_from_context",
]
