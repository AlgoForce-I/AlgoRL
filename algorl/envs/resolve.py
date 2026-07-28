"""Resolve user-provided environments into :class:`~algorl.envs.training_env.TrainingEnv`."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym

from algorl.agents.configs import BaseAgentConfig, SearchAgentConfig
from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv
from algorl.envs.training_env import TrainingEnv

EnvSource = (
    str
    | gym.Env
    | gym.vector.VectorEnv
    | JaxEnv
    | BatchedJaxEnv
    | TrainingEnv
    | Callable[[], gym.Env]
)


def _as_gym_env_factory(env: EnvSource) -> Callable[[], gym.Env]:
    if isinstance(env, str):
        return lambda: gym.make(env)
    if callable(env):
        return env
    if isinstance(env, gym.Env):
        if env.spec is not None and env.spec.id:
            env_id = env.spec.id

            def make_from_id() -> gym.Env:
                return gym.make(env_id)

            return make_from_id
        raise TypeError(
            "Pass a Gymnasium env id string or a factory ``lambda: gym.make(...)`` "
            f"for automatic vectorization; got bare {type(env)!r} without env.spec.id."
        )
    raise TypeError(
        f"Expected gymnasium env id, gym.Env, VectorEnv, JaxEnv, TrainingEnv, or env factory; "
        f"got {type(env)!r}."
    )


def _ensure_search_wrapper(env: gym.Env) -> gym.Env:
    if hasattr(env, "search_environment"):
        return env
    from algorl.backends.jax.envs.gymnasium_env import GymnasiumSearchEnv

    return GymnasiumSearchEnv(env)


def _target_rollout_envs(
    config: BaseAgentConfig | None,
    *,
    num_envs: int | None,
) -> int:
    if num_envs is not None:
        return max(1, num_envs)
    if isinstance(config, SearchAgentConfig) and config.uses_batched_rollout:
        return config.rollout_envs
    return 1


def _validate_env_config(
    env: TrainingEnv,
    config: BaseAgentConfig | None,
    *,
    expected_envs: int | None = None,
) -> TrainingEnv:
    if config is None or not isinstance(config, SearchAgentConfig):
        return env
    if not config.uses_batched_rollout:
        return env
    expected = expected_envs if expected_envs is not None else config.rollout_envs
    if env.num_envs != expected:
        raise ValueError(
            f"Config expects {expected} parallel rollout env(s) "
            f"(search_batch_size={config.search_batch_size}), "
            f"but the training environment has num_envs={env.num_envs}."
        )
    return env


def resolve_env(env: object, *, seed: int = 0) -> TrainingEnv:
    """Normalize Gymnasium, JAX-native, or MTCWorldMJX environments for training."""
    if isinstance(env, TrainingEnv):
        return env

    if isinstance(env, gym.vector.VectorEnv):
        from algorl.backends.jax.envs.gymnasium_vector import GymnasiumVectorJaxEnv

        return TrainingEnv.from_jax(GymnasiumVectorJaxEnv(env, seed=seed), seed=seed)

    if isinstance(env, gym.Env):
        return TrainingEnv.from_gymnasium(env)

    if isinstance(env, (JaxEnv, BatchedJaxEnv)):
        return TrainingEnv.from_jax(env, seed=seed)

    from algorl.backends.jax.envs.mtcworld_jax import as_mtcworld_jax_env

    mtc_env = as_mtcworld_jax_env(env)
    if mtc_env is not None:
        return TrainingEnv.from_jax(mtc_env, seed=seed)

    raise TypeError(
        f"Expected gymnasium.Env, gymnasium.vector.VectorEnv, JaxEnv, BatchedJaxEnv, "
        f"MTCWorldMJX env, or TrainingEnv; got {type(env)!r}."
    )


def make_env(
    env: EnvSource,
    *,
    config: BaseAgentConfig | None = None,
    seed: int = 0,
    num_envs: int | None = None,
) -> TrainingEnv:
    """Build a :class:`TrainingEnv` from common user inputs.

    Accepts a Gymnasium env id (``"HalfCheetah-v5"``), a ``gym.make(...)`` env,
    a vector/JAX env, or a factory ``lambda: gym.make(...)``.

    When ``config`` is a batched search preset (``search_batch_size > 1``), a
    plain Gymnasium env is automatically wrapped in a vectorized JAX rollout env
    with that many lanes. Sequential presets wrap single envs for MCTS search.
    """
    if isinstance(env, TrainingEnv):
        return _validate_env_config(env, config, expected_envs=num_envs)

    if isinstance(env, (JaxEnv, BatchedJaxEnv, gym.vector.VectorEnv)):
        resolved = resolve_env(env, seed=seed)
        return _validate_env_config(
            resolved,
            config,
            expected_envs=num_envs or resolved.num_envs,
        )

    from algorl.backends.jax.envs.mtcworld_jax import as_mtcworld_jax_env

    mtc_env = as_mtcworld_jax_env(env)
    if mtc_env is not None:
        resolved = TrainingEnv.from_jax(mtc_env, seed=seed)
        return _validate_env_config(
            resolved,
            config,
            expected_envs=num_envs or resolved.num_envs,
        )

    rollout_envs = _target_rollout_envs(config, num_envs=num_envs)
    factory = _as_gym_env_factory(env)

    if rollout_envs > 1:
        from algorl.backends.jax.envs.gymnasium_vector import make_gymnasium_vector_env

        vector_env = make_gymnasium_vector_env(factory, rollout_envs, seed=seed)
        resolved = resolve_env(vector_env, seed=seed)
        return _validate_env_config(resolved, config, expected_envs=rollout_envs)

    gym_env = _ensure_search_wrapper(factory())
    return resolve_env(gym_env, seed=seed)
