"""MTCWorldMJX native objects as :class:`~algorl.envs.jax_env.JaxEnv` adapters."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.backends.jax.envs.mtcworld_search import (
    MtcworldCWSearchEnvironment,
    MtcworldCWTaskSearchEnvironment,
    MtcworldSearchEnvironment,
)
from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv, JaxRolloutBatch, JaxState, PolicyFn


def require_mtcworld() -> Any:
    try:
        import MTCWorldMJX as mtc
    except ImportError as exc:
        raise ImportError(
            "MTCWorldMJX is not installed. Install the optional JAX extra, e.g. "
            "``pip install -e '.[jax,mtcworld]'`` or ``pip install -e /path/to/ContinualWorldMJX``."
        ) from exc
    return mtc


def _as_bool(value: jnp.ndarray | bool | float) -> bool:
    return bool(np.asarray(value) > 0)


def _as_float(value: jnp.ndarray | float) -> float:
    return float(np.asarray(value))


class SawyerJaxEnv(JaxEnv):
    """Adapter for ``MTCWorldMJX`` ``SawyerXYZEnv``."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self._search_env = MtcworldSearchEnvironment(env)
        self._jit_reset = jax.jit(env.reset)
        self._jit_step = jax.jit(env.step)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self._env.observation_size,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self._env.action_size,)

    def search_environment(self) -> SearchEnvironment:
        return self._search_env

    def reset(self, key: jnp.ndarray) -> JaxState:
        state = self._jit_reset(key)
        self._search_env.bind_state(state)
        return state

    def step(self, state: JaxState, action: jnp.ndarray) -> JaxState:
        next_state = self._jit_step(state, action)
        self._search_env.bind_state(next_state)
        return next_state

    def observation(self, state: JaxState) -> jnp.ndarray:
        return jnp.asarray(state.obs, dtype=jnp.float32)

    def reward(self, state: JaxState) -> float:
        return _as_float(state.reward)

    def terminated(self, state: JaxState) -> bool:
        return _as_bool(state.terminated)

    def truncated(self, state: JaxState) -> bool:
        return _as_bool(state.truncated)

    def info(self, state: JaxState) -> dict[str, Any]:
        del state
        return {}

    def reset_after_episode(self, key: jnp.ndarray, state: JaxState) -> JaxState:
        del state
        return self.reset(key)


class CWTaskJaxEnv(JaxEnv):
    """Adapter for ``MTCWorldMJX.cw_env.CWTaskEnv``."""

    def __init__(self, task_env: Any) -> None:
        self._task_env = task_env
        self._search_env = MtcworldCWTaskSearchEnvironment(task_env)
        self._jit_reset = jax.jit(task_env.reset)
        self._jit_step = jax.jit(task_env.step)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (int(self._task_env.num_tasks) + 39,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self._task_env.env.action_size,)

    @property
    def task_env(self) -> Any:
        return self._task_env

    def search_environment(self) -> SearchEnvironment:
        return self._search_env

    def reset(self, key: jnp.ndarray) -> JaxState:
        state = self._jit_reset(key)
        self._search_env.bind_state(state)
        return state

    def step(self, state: JaxState, action: jnp.ndarray) -> JaxState:
        next_state = self._jit_step(state, action)
        self._search_env.bind_state(next_state)
        return next_state

    def observation(self, state: JaxState) -> jnp.ndarray:
        return jnp.asarray(state.obs, dtype=jnp.float32)

    def reward(self, state: JaxState) -> float:
        return _as_float(state.reward)

    def terminated(self, state: JaxState) -> bool:
        return _as_bool(state.terminated)

    def truncated(self, state: JaxState) -> bool:
        return _as_bool(state.truncated)

    def info(self, state: JaxState) -> dict[str, Any]:
        return {
            "task_index": int(self._task_env.task_idx),
            "task_name": self._task_env.env_name,
            "success": _as_float(state.metrics.get("success", 0.0)),
        }

    def reset_after_episode(self, key: jnp.ndarray, state: JaxState) -> JaxState:
        del state
        return self.reset(key)


class ContinualLearningJaxEnv(JaxEnv):
    """Adapter for ``MTCWorldMJX.cw_env.ContinualLearningEnv``."""

    def __init__(self, cl_env: Any) -> None:
        self._cl_env = cl_env
        self._search_env = MtcworldCWSearchEnvironment(cl_env)
        self._jit_reset = jax.jit(cl_env.reset)
        self._jit_step = jax.jit(cl_env.step)
        self._jit_reset_from_state = jax.jit(cl_env.reset_from_state)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        mtc = require_mtcworld()
        return (mtc.cw_obs_dim(self._cl_env.num_tasks),)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self._cl_env.task_envs[0].env.action_size,)

    @property
    def continual_env(self) -> Any:
        return self._cl_env

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(task.env_name for task in self._cl_env.task_envs)

    def search_environment(self) -> SearchEnvironment:
        return self._search_env

    def reset(self, key: jnp.ndarray) -> JaxState:
        state = self._jit_reset(key)
        self._search_env.bind_state(state)
        return state

    def step(self, state: JaxState, action: jnp.ndarray) -> JaxState:
        next_state = self._jit_step(state, action)
        self._search_env.bind_state(next_state)
        return next_state

    def observation(self, state: JaxState) -> jnp.ndarray:
        return jnp.asarray(state.obs, dtype=jnp.float32)

    def reward(self, state: JaxState) -> float:
        return _as_float(state.reward)

    def terminated(self, state: JaxState) -> bool:
        return _as_bool(state.terminated)

    def truncated(self, state: JaxState) -> bool:
        return _as_bool(state.truncated)

    def info(self, state: JaxState) -> dict[str, Any]:
        seq_idx = int(state.info["seq_idx"])
        return {
            "seq_idx": seq_idx,
            "global_step": int(state.info["global_step"]),
            "task_changed": bool(state.info.get("task_changed", False)),
            "forced_task_change": bool(state.info.get("forced_task_change", False)),
            "task_name": self.task_names[seq_idx],
            "success": _as_float(state.metrics.get("success", 0.0)),
        }

    def reset_after_episode(self, key: jnp.ndarray, state: JaxState) -> JaxState:
        if int(state.info["global_step"]) >= self._cl_env.steps_limit:
            return self.reset(key)
        next_state = self._jit_reset_from_state(key, state)
        self._search_env.bind_state(next_state)
        return next_state


class VectorJaxEnv(BatchedJaxEnv):
    """Adapter for ``MTCWorldMJX.mt_benchmarks.VectorEnv``."""

    def __init__(
        self,
        vector_env: Any,
        *,
        task_index: int | None = None,
        num_tasks: int | None = None,
        append_one_hot: bool = False,
    ) -> None:
        self._vector_env = vector_env
        self._task_index = task_index
        self._num_tasks = num_tasks
        if append_one_hot:
            if task_index is None or num_tasks is None:
                raise ValueError("append_one_hot requires task_index and num_tasks")
            from MTCWorldMJX.cw_env import append_task_one_hot_batched

            self._append_one_hot_fn = append_task_one_hot_batched
        else:
            self._append_one_hot_fn = None

    @property
    def num_envs(self) -> int:
        return self._vector_env.num_envs

    @property
    def vector_env(self) -> Any:
        return self._vector_env

    @property
    def observation_shape(self) -> tuple[int, ...]:
        obs_dim = int(self._vector_env.env.observation_size)
        if self._append_one_hot_fn is not None and self._num_tasks is not None:
            obs_dim += self._num_tasks
        return (obs_dim,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self._vector_env.env.action_size,)

    def reset(self, key: jnp.ndarray) -> JaxState:
        return self._vector_env.reset(key)

    def step(self, state: JaxState, actions: jnp.ndarray) -> JaxState:
        return self._vector_env.step(state, actions)

    def observation(self, state: JaxState) -> jnp.ndarray:
        obs = jnp.asarray(state.obs, dtype=jnp.float32)
        if self._append_one_hot_fn is not None:
            assert self._task_index is not None
            assert self._num_tasks is not None
            return self._append_one_hot_fn(obs, self._task_index, self._num_tasks)
        return obs

    def collect_rollout(
        self,
        policy: PolicyFn,
        num_steps: int,
        *,
        key: jnp.ndarray,
    ) -> JaxRolloutBatch:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        reset_key, rollout_key = jax.random.split(key)
        state = self.reset(reset_key)

        def body(carry: JaxState, step_key: jnp.ndarray) -> tuple[JaxState, dict[str, jnp.ndarray]]:
            current_state = carry
            obs = self.observation(current_state)
            actions = policy(obs, step_key)
            next_state = self.step(current_state, actions)
            next_obs = self.observation(next_state)
            done = (next_state.truncated + next_state.terminated) > 0
            return next_state, {
                "observation": obs,
                "action": actions,
                "reward": next_state.reward,
                "next_observation": next_obs,
                "done": done,
            }

        step_keys = jax.random.split(rollout_key, num_steps)
        final_state, trajectory = jax.lax.scan(body, state, step_keys)
        del final_state

        return JaxRolloutBatch(
            observation=np.asarray(trajectory["observation"], dtype=np.float32),
            action=np.asarray(trajectory["action"], dtype=np.float32),
            reward=np.asarray(trajectory["reward"], dtype=np.float32),
            next_observation=np.asarray(trajectory["next_observation"], dtype=np.float32),
            done=np.asarray(trajectory["done"], dtype=bool),
        )


def as_mtcworld_jax_env(env: object) -> JaxEnv | BatchedJaxEnv | None:
    """Return a JAX adapter when ``env`` is a native MTCWorldMJX object."""
    try:
        from MTCWorldMJX.cw_env import ContinualLearningEnv, CWTaskEnv
        from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZEnv
        from MTCWorldMJX.mt_benchmarks import VectorEnv
    except ImportError:
        return None

    if isinstance(env, SawyerXYZEnv):
        return SawyerJaxEnv(env)
    if isinstance(env, ContinualLearningEnv):
        return ContinualLearningJaxEnv(env)
    if isinstance(env, CWTaskEnv):
        return CWTaskJaxEnv(env)
    if isinstance(env, VectorEnv):
        return VectorJaxEnv(env)
    return None


def as_mtcworld_jax_env_from_spec(
    *,
    benchmark: str | None = None,
    env_id: str | None = None,
    task_index: int | None = None,
    vector_env: Any | None = None,
    num_envs: int = 1,
    seed: int = 0,
    config: Any | None = None,
    steps_per_task: int | None = None,
) -> JaxEnv | BatchedJaxEnv:
    """Build a native MTCWorld object and return its JAX adapter."""
    mtc = require_mtcworld()

    if vector_env is not None:
        if benchmark is not None and task_index is not None:
            bench = mtc.CWBenchmark(benchmark, config=config, seed=seed)
            return VectorJaxEnv(
                vector_env,
                task_index=task_index,
                num_tasks=bench.num_tasks,
                append_one_hot=True,
            )
        return VectorJaxEnv(vector_env)

    if benchmark is not None:
        if task_index is not None:
            test_envs = mtc.make_cl_test_envs(benchmark, config=config, seed=seed)
            return CWTaskJaxEnv(test_envs[task_index])
        cl_env = mtc.make_cl_train_env(
            benchmark,
            config=config,
            seed=seed,
            steps_per_task=steps_per_task,
        )
        return ContinualLearningJaxEnv(cl_env)

    if env_id is not None:
        if num_envs > 1:
            vector = mtc.make_mt_envs(env_id, seed=seed, num_envs=num_envs, config=config)
            if not hasattr(vector, "reset"):
                raise TypeError(f"make_mt_envs({env_id!r}) did not return a VectorEnv.")
            return VectorJaxEnv(vector)
        return SawyerJaxEnv(mtc.make(env_id, config=config))

    raise ValueError("Provide env_id, benchmark, or vector_env.")


MtcworldRolloutBatch = JaxRolloutBatch


class MtcworldRolloutCollector:
    """JIT vectorized rollout over one MTCWorldMJX task."""

    def __init__(
        self,
        env_id: str,
        *,
        num_envs: int = 8,
        seed: int = 0,
        config: Any | None = None,
    ) -> None:
        adapter = as_mtcworld_jax_env_from_spec(
            env_id=env_id,
            num_envs=num_envs,
            seed=seed,
            config=config,
        )
        if not isinstance(adapter, VectorJaxEnv):
            raise TypeError(f"Expected VectorJaxEnv for rollout collection, got {type(adapter)!r}.")
        self.env_id = env_id
        self.num_envs = adapter.num_envs
        self.seed = seed
        self._adapter = adapter

    @property
    def vector_env(self) -> Any:
        return self._adapter.vector_env

    def collect_rollout(
        self,
        policy: PolicyFn,
        num_steps: int,
        *,
        key: Any | None = None,
    ) -> MtcworldRolloutBatch:
        rollout_key = key if key is not None else jax.random.PRNGKey(self.seed)
        return self._adapter.collect_rollout(policy, num_steps, key=rollout_key)


class MtcworldCWRolloutCollector(MtcworldRolloutCollector):
    """Vectorized rollout for one Continual World sequence slot."""

    def __init__(
        self,
        benchmark: str,
        task_index: int,
        *,
        num_envs: int = 8,
        seed: int = 0,
        config: Any | None = None,
    ) -> None:
        from MTCWorldMJX.cw_benchmarks import CWBenchmark, cw_sawyer_config
        from MTCWorldMJX.mt_benchmarks import VectorEnv, rand_vecs_for_env

        bench = CWBenchmark(benchmark, config=config, seed=seed)
        env_name = bench.task_names[task_index]
        sawyer_config = cw_sawyer_config(bench.config)
        vector_env = VectorEnv(
            env_name,
            rand_vecs_for_env(bench.tasks, env_name),
            num_envs,
            config=sawyer_config,
            partially_observable=bench.config.partially_observable,
            task_select="random",
            seed=seed,
        )
        adapter = as_mtcworld_jax_env_from_spec(
            benchmark=benchmark,
            task_index=task_index,
            vector_env=vector_env,
            seed=seed,
            config=config,
        )
        if not isinstance(adapter, VectorJaxEnv):
            raise TypeError(f"Expected VectorJaxEnv, got {type(adapter)!r}.")
        self.benchmark_name = benchmark
        self.task_index = task_index
        self.env_id = env_name
        self.num_envs = adapter.num_envs
        self.seed = seed
        self._adapter = adapter


class MtcworldContinualRolloutCollector:
    """Run batched rollouts across every task in a CW10/CW20 sequence."""

    def __init__(
        self,
        benchmark: str = "CW10",
        *,
        num_envs: int = 8,
        seed: int = 0,
        config: Any | None = None,
    ) -> None:
        from MTCWorldMJX.cw_benchmarks import CWBenchmark

        bench = CWBenchmark(benchmark, config=config, seed=seed)
        self.benchmark_name = benchmark
        self.num_tasks = bench.num_tasks
        self.task_names = tuple(bench.task_names)
        self._collectors = [
            MtcworldCWRolloutCollector(
                benchmark,
                task_index,
                num_envs=num_envs,
                seed=seed + task_index,
                config=config,
            )
            for task_index in range(bench.num_tasks)
        ]

    def collect_sequence(
        self,
        policy: PolicyFn,
        steps_per_task: int,
        *,
        key: Any | None = None,
    ) -> list[MtcworldRolloutBatch]:
        rollout_key = key if key is not None else jax.random.PRNGKey(0)
        batches: list[MtcworldRolloutBatch] = []
        for collector in self._collectors:
            rollout_key, step_key = jax.random.split(rollout_key)
            batches.append(collector.collect_rollout(policy, steps_per_task, key=step_key))
        return batches
