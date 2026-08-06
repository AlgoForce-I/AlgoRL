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
from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv, JaxRolloutBatch, JaxState, PolicyFn, RolloutStepCallback


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


def _emit_rollout_step(
    on_step: RolloutStepCallback | None,
    *,
    num_envs: int,
    reward: np.ndarray,
    done: np.ndarray,
    lane_infos: list[dict[str, Any]] | None = None,
    step_info: dict[str, Any] | None = None,
) -> None:
    if on_step is None:
        return
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    done_array = np.asarray(done, dtype=bool).reshape(-1)
    infos = lane_infos if lane_infos is not None else [{} for _ in range(int(reward_array.shape[0]))]
    info: dict[str, Any] = {
        "train/reward": float(np.mean(reward_array)),
        "rewards": reward_array,
        "dones": done_array,
        "infos": [dict(item) for item in infos],
    }
    if step_info is not None:
        info.update(step_info)
    on_step(int(reward_array.shape[0]), info)


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
        state = self._cl_env.reset(key)
        self._search_env.bind_state(state)
        return state

    def step(self, state: JaxState, action: jnp.ndarray) -> JaxState:
        next_state = self._cl_env.step(state, action)
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
        next_state = self._cl_env.reset_from_state(key, state)
        self._search_env.bind_state(next_state)
        return next_state


def _build_cw_vector_jax_env(
    benchmark: str,
    task_index: int,
    *,
    num_envs: int,
    seed: int,
    config: Any | None,
    bench: Any | None = None,
    autoreset: bool = True,
) -> VectorJaxEnv:
    """Vectorized JAX env for one Continual World task with task one-hot observations."""
    from MTCWorldMJX.cw_benchmarks import CWBenchmark, cw_sawyer_config
    from MTCWorldMJX.mt_benchmarks import VectorEnv, rand_vecs_for_env

    resolved_bench = bench or CWBenchmark(benchmark, config=config, seed=seed)
    env_name = resolved_bench.task_names[task_index]
    sawyer_config = cw_sawyer_config(resolved_bench.config)
    vector_env = VectorEnv(
        env_name,
        rand_vecs_for_env(resolved_bench.tasks, env_name),
        num_envs,
        config=sawyer_config,
        partially_observable=resolved_bench.config.partially_observable,
        task_select="random",
        autoreset=autoreset,
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
    return adapter


class BatchedContinualLearningJaxEnv:
    """Parallel rollouts within each CW task; tasks advance sequentially."""

    def __init__(
        self,
        benchmark: str,
        *,
        num_envs: int,
        seed: int,
        config: Any | None = None,
        steps_per_task: int,
        bench: Any | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if steps_per_task < 1:
            raise ValueError("steps_per_task must be >= 1")

        from MTCWorldMJX.cw_benchmarks import CWBenchmark

        self._bench = bench or CWBenchmark(benchmark, config=config, seed=seed)
        self.benchmark_name = benchmark
        self.num_envs = num_envs
        self.seed = seed
        self.config = config
        self.steps_per_task = steps_per_task
        self.num_tasks = self._bench.num_tasks
        self.task_names = tuple(self._bench.task_names)
        self.steps_limit = self.num_tasks * steps_per_task
        mtc = require_mtcworld()
        self._obs_dim = mtc.cw_obs_dim(self.num_tasks)

        self._seq_idx = 0
        self._global_step = 0
        self._vector_env = self._make_task_vector_env(0)
        self._state: JaxState | None = None

    def _make_task_vector_env(self, task_index: int) -> VectorJaxEnv:
        # Explicit resets (instead of VectorEnv's on-device autoreset) so that
        # every episode resamples goals from the task pool (CW random_init_all)
        # and the true terminal observation is available to the replay buffer.
        return _build_cw_vector_jax_env(
            self.benchmark_name,
            task_index,
            num_envs=self.num_envs,
            seed=self.seed + task_index,
            config=self.config,
            bench=self._bench,
            autoreset=False,
        )

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self._obs_dim,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._vector_env.action_shape

    @property
    def current_task_index(self) -> int:
        return self._seq_idx

    @property
    def current_task_name(self) -> str:
        return self.task_names[self._seq_idx]

    def curriculum_checkpoint_state(self) -> dict[str, Any]:
        """Persist CW task index / global step (training-exact resume)."""
        return {
            "seq_idx": int(self._seq_idx),
            "global_step": int(self._global_step),
            "steps_per_task": int(self.steps_per_task),
            "num_tasks": int(self.num_tasks),
            "seed": int(self.seed),
        }

    def load_curriculum_checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        key: jnp.ndarray,
    ) -> None:
        """Restore curriculum counters and rebuild the active task vector env."""
        if int(state.get("steps_per_task", self.steps_per_task)) != int(self.steps_per_task):
            raise ValueError(
                "steps_per_task mismatch on env resume: "
                f"checkpoint={state.get('steps_per_task')} live={self.steps_per_task}."
            )
        self._seq_idx = int(state["seq_idx"])
        self._global_step = int(state["global_step"])
        if self._seq_idx < 0 or self._seq_idx >= self.num_tasks:
            raise ValueError(
                f"Invalid seq_idx={self._seq_idx} for num_tasks={self.num_tasks}."
            )
        self._vector_env = self._make_task_vector_env(self._seq_idx)
        self._state = self._vector_env.reset(key)

    def reset(self, key: jnp.ndarray) -> JaxState:
        self._seq_idx = 0
        self._global_step = 0
        self._vector_env = self._make_task_vector_env(0)
        self._state = self._vector_env.reset(key)
        return self._state

    def observation(self, state: JaxState) -> jnp.ndarray:
        return self._vector_env.observation(state)

    def step(self, state: JaxState, actions: jnp.ndarray) -> JaxState:
        return self._vector_env.step(state, actions)

    def _advance_task(self, key: jnp.ndarray) -> JaxState:
        self._seq_idx += 1
        del self._vector_env
        self._vector_env = self._make_task_vector_env(self._seq_idx)
        key, reset_key = jax.random.split(key)
        return self._vector_env.reset(reset_key)

    def _lane_info(
        self,
        state: JaxState,
        env_idx: int,
        *,
        seq_idx: int,
        task_changed: bool,
        forced_task_change: bool,
    ) -> dict[str, Any]:
        success = state.metrics.get("success", jnp.zeros((self.num_envs,)))
        return {
            "seq_idx": seq_idx,
            "global_step": self._global_step,
            "task_name": self.task_names[seq_idx],
            "task_changed": task_changed,
            "forced_task_change": forced_task_change,
            "success": _as_float(success[env_idx]),
        }

    def collect_rollout(
        self,
        policy: PolicyFn,
        num_steps: int,
        *,
        key: jnp.ndarray,
        on_step: RolloutStepCallback | None = None,
    ) -> JaxRolloutBatch:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self._state is None:
            key, reset_key = jax.random.split(key)
            self.reset(reset_key)

        _, rollout_key = jax.random.split(key)
        step_keys = jax.random.split(rollout_key, num_steps)
        state = self._state
        assert state is not None

        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        step_infos: list[list[dict[str, Any]]] = []

        for step_idx in range(num_steps):
            obs = np.asarray(self.observation(state), dtype=np.float32)
            action = np.asarray(policy(obs, step_keys[step_idx]), dtype=np.float32)
            stepped_state = self.step(state, jnp.asarray(action, dtype=jnp.float32))
            # Capture the stepped transition before any reset/advance replaces
            # the state, so the recorded reward, terminal observation, and
            # success metric belong to this step rather than a reset state.
            next_state = stepped_state
            next_obs = np.asarray(self.observation(stepped_state), dtype=np.float32)
            reward = np.asarray(stepped_state.reward, dtype=np.float32)
            episode_done = np.asarray(
                (stepped_state.truncated + stepped_state.terminated) > 0,
                dtype=bool,
            )

            self._global_step += self.num_envs
            step_seq_idx = self._seq_idx
            task_end = (self._seq_idx + 1) * self.steps_per_task
            forced_task_change = self._global_step >= task_end
            task_changed = forced_task_change and self._seq_idx < self.num_tasks - 1

            if task_changed:
                _, advance_key = jax.random.split(step_keys[step_idx])
                next_state = self._advance_task(advance_key)
                done = np.ones((self.num_envs,), dtype=bool)
            else:
                done = episode_done
                if forced_task_change and self._seq_idx >= self.num_tasks - 1:
                    done = np.ones((self.num_envs,), dtype=bool)
                if episode_done.any():
                    # CW episodes are truncation-only with a shared horizon, so
                    # all lanes finish together. Reset explicitly to resample
                    # goals from the task pool (random_init_all protocol).
                    if not episode_done.all():
                        raise RuntimeError(
                            "CW lanes desynchronized: expected all lanes to "
                            "truncate on the same step."
                        )
                    _, episode_key = jax.random.split(step_keys[step_idx])
                    next_state = self._vector_env.reset(episode_key)

            step_infos.append(
                [
                    self._lane_info(
                        stepped_state,
                        env_idx,
                        seq_idx=step_seq_idx,
                        task_changed=task_changed,
                        forced_task_change=forced_task_change,
                    )
                    for env_idx in range(self.num_envs)
                ]
            )
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(done)
            state = next_state
            lane_info = step_infos[-1][0]
            _emit_rollout_step(
                on_step,
                num_envs=self.num_envs,
                reward=rewards[-1],
                done=done,
                lane_infos=step_infos[-1],
                step_info={
                    "task_name": lane_info.get("task_name"),
                    "train/episode_success": lane_info.get("success"),
                },
            )

        self._state = state
        return JaxRolloutBatch(
            observation=np.stack(observations, axis=0),
            action=np.stack(actions, axis=0),
            reward=np.stack(rewards, axis=0),
            next_observation=np.stack(next_observations, axis=0),
            done=np.stack(dones, axis=0),
            step_info=step_infos,
        )


def make_batched_cw_train_env(
    benchmark: str,
    *,
    num_envs: int = 8,
    seed: int = 0,
    config: Any | None = None,
    steps_per_task: int | None = None,
) -> BatchedContinualLearningJaxEnv:
    """Batched CW training: ``num_envs`` parallel actors per task, tasks in sequence."""
    from MTCWorldMJX.cw_benchmarks import CWBenchmark

    bench = CWBenchmark(benchmark, config=config, seed=seed)
    resolved_steps = steps_per_task
    if resolved_steps is None and config is not None:
        resolved_steps = getattr(config, "steps_per_task", None)
    if resolved_steps is None:
        resolved_steps = bench.config.steps_per_task
    return BatchedContinualLearningJaxEnv(
        benchmark,
        num_envs=num_envs,
        seed=seed,
        config=config,
        steps_per_task=int(resolved_steps),
        bench=bench,
    )


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
        on_step: RolloutStepCallback | None = None,
    ) -> JaxRolloutBatch:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        reset_key, rollout_key = jax.random.split(key)
        state = self.reset(reset_key)
        step_keys = jax.random.split(rollout_key, num_steps)

        # Eager loop: policies may call MCTS / planner code with NumPy and Python
        # side effects, which cannot run inside ``jax.lax.scan``.
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []
        dones: list[np.ndarray] = []

        for step_idx in range(num_steps):
            obs = np.asarray(self.observation(state), dtype=np.float32)
            action = np.asarray(policy(obs, step_keys[step_idx]), dtype=np.float32)
            next_state = self.step(state, jnp.asarray(action, dtype=jnp.float32))
            next_obs = np.asarray(self.observation(next_state), dtype=np.float32)
            done = np.asarray(
                (next_state.truncated + next_state.terminated) > 0,
                dtype=bool,
            )

            observations.append(obs)
            actions.append(action)
            rewards.append(np.asarray(next_state.reward, dtype=np.float32))
            next_observations.append(next_obs)
            dones.append(done)
            state = next_state
            _emit_rollout_step(
                on_step,
                num_envs=self.num_envs,
                reward=rewards[-1],
                done=done,
            )

        return JaxRolloutBatch(
            observation=np.stack(observations, axis=0),
            action=np.stack(actions, axis=0),
            reward=np.stack(rewards, axis=0),
            next_observation=np.stack(next_observations, axis=0),
            done=np.stack(dones, axis=0),
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
    def jax_env(self) -> VectorJaxEnv:
        """Batched JAX adapter for :class:`~algorl.envs.training_env.TrainingEnv.from_jax`."""
        return self._adapter

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
        adapter = _build_cw_vector_jax_env(
            benchmark,
            task_index,
            num_envs=num_envs,
            seed=seed,
            config=config,
        )
        self.benchmark_name = benchmark
        self.task_index = task_index
        self.env_id = adapter.vector_env.env_name
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
