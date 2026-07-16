"""Resolve user-provided environments into :class:`~algorl.envs.training_env.TrainingEnv`."""

from __future__ import annotations

import gymnasium as gym

from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv
from algorl.envs.training_env import TrainingEnv


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
