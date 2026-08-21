"""Periodic evaluation rollouts isolated from the training environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from algorl.common.episode_metrics import sanitize_task_name
from algorl.common.logger import Logger
from algorl.common.progress_bar import TqdmProgressBar
from algorl.core.learner import Learner
from algorl.core.planner import BatchedPlanner, Planner
from algorl.envs.training_env import TrainingEnv

EVAL_EPISODES = 3
DEFAULT_EVAL_HORIZON = 10_000
CW_EVAL_HORIZON = 200
CW_BENCHMARK_NAMES = ("CW10", "CW20")


class EvalVectorEnv(Protocol):
    """Vectorized eval env with ``num_envs`` parallel episode lanes."""

    num_envs: int

    def reset(self, *, seed: int | None = None) -> np.ndarray: ...

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EvalTask:
    """One evaluation task (a CW slot, or the single train task)."""

    name: str
    task_index: int | None = None
    kind: str = "custom"
    env_id: str | None = None
    benchmark: str | None = None
    cw_config: Any = None
    max_steps: int = DEFAULT_EVAL_HORIZON


EvalEnvFactory = Callable[[EvalTask], EvalVectorEnv]


def should_evaluate(
    completed_steps: int,
    *,
    period: int,
    last_eval_bucket: int,
    last_eval_step: int,
    total_timesteps: int,
    at_end: bool = False,
) -> bool:
    """Return whether an eval should run after ``completed_steps`` env steps."""
    if period <= 0 or completed_steps <= 0:
        return False
    if completed_steps < period:
        return False
    bucket = completed_steps // period
    if bucket > last_eval_bucket:
        return True
    return bool(
        at_end
        and completed_steps >= int(total_timesteps)
        and last_eval_step < completed_steps
    )


def discover_eval_tasks(train_env: TrainingEnv) -> tuple[EvalTask, ...]:
    """Infer eval tasks from a training env without using agent config."""
    for obj in _env_objects(train_env):
        benchmark = getattr(obj, "benchmark_name", None)
        num_tasks = getattr(obj, "num_tasks", None)
        if not (
            isinstance(benchmark, str)
            and benchmark in CW_BENCHMARK_NAMES
            and isinstance(num_tasks, int)
            and num_tasks >= 1
        ):
            continue
        names = getattr(obj, "task_names", None)
        task_names = (
            tuple(str(name) for name in names)
            if names is not None
            else tuple(f"task_{index}" for index in range(num_tasks))
        )
        config = getattr(obj, "config", None)
        return tuple(
            EvalTask(
                name=task_names[index] if index < len(task_names) else f"task_{index}",
                task_index=index,
                kind="cw",
                benchmark=benchmark,
                cw_config=config,
                max_steps=CW_EVAL_HORIZON,
            )
            for index in range(num_tasks)
        )

    env_id = _mtcworld_single_task_id(train_env)
    if env_id is not None:
        return (
            EvalTask(
                name=env_id,
                kind="mtcworld",
                env_id=env_id,
                max_steps=CW_EVAL_HORIZON,
            ),
        )

    gym_id = _gym_env_id(train_env)
    if gym_id is not None:
        horizon = _gym_max_episode_steps(train_env) or DEFAULT_EVAL_HORIZON
        return (
            EvalTask(
                name=gym_id,
                kind="gym",
                env_id=gym_id,
                max_steps=horizon,
            ),
        )

    raise TypeError(
        "Cannot infer an evaluation environment from the training env. "
        "Pass a Gymnasium env with spec.id, a Continual World env, or an "
        "eval_env_factory."
    )


class PeriodicEvaluator:
    """Run ``EVAL_EPISODES`` parallel eval episodes on a period of env steps."""

    def __init__(
        self,
        *,
        period: int,
        train_env: TrainingEnv,
        planner: Planner,
        learner: Learner,
        logger: Logger,
        seed: int,
        start_step: int = 0,
        num_episodes: int = EVAL_EPISODES,
        env_factory: EvalEnvFactory | None = None,
        tasks: tuple[EvalTask, ...] | list[EvalTask] | None = None,
    ) -> None:
        if period <= 0:
            raise ValueError(f"eval period must be > 0, got {period}.")
        self.period = int(period)
        self.train_env = train_env
        self.planner = planner
        self.learner = learner
        self.logger = logger
        self.seed = int(seed)
        self.num_episodes = max(1, int(num_episodes))
        self._env_factory = env_factory
        self._eval_count = 0
        completed = max(0, int(start_step))
        self._last_eval_step = completed
        self._last_eval_bucket = completed // self.period
        if tasks is not None:
            self.tasks = tuple(tasks)
        elif env_factory is not None:
            try:
                self.tasks = discover_eval_tasks(train_env)
            except TypeError:
                self.tasks = (
                    EvalTask(name="default", kind="custom", max_steps=DEFAULT_EVAL_HORIZON),
                )
        else:
            self.tasks = discover_eval_tasks(train_env)
        self._cached_envs: dict[str, EvalVectorEnv] = {}

    def maybe_run(
        self,
        completed_steps: int,
        *,
        total_timesteps: int,
        at_end: bool = False,
        progress_bar: TqdmProgressBar | None = None,
    ) -> dict[str, float] | None:
        """Run eval if the period elapsed; log metrics at the current train step."""
        if not should_evaluate(
            int(completed_steps),
            period=self.period,
            last_eval_bucket=self._last_eval_bucket,
            last_eval_step=self._last_eval_step,
            total_timesteps=int(total_timesteps),
            at_end=at_end,
        ):
            return None
        metrics = self.run(progress_bar=progress_bar)
        log_step = max(0, int(completed_steps) - 1)
        self.logger.record(log_step, metrics)
        flush = getattr(self.logger, "flush", None)
        if callable(flush):
            flush()
        self._last_eval_step = int(completed_steps)
        self._last_eval_bucket = int(completed_steps) // self.period
        self._eval_count += 1
        if progress_bar is not None:
            progress_bar.pulse({**metrics, "phase": "eval"})
        return metrics

    def run(self, *, progress_bar: TqdmProgressBar | None = None) -> dict[str, float]:
        """Evaluate every task; does not step the training env or write replay."""
        last_result = getattr(self.planner, "last_result", _MISSING)
        try:
            task_results: list[tuple[EvalTask, list[_EpisodeStats]]] = []
            for task_offset, task in enumerate(self.tasks):
                episodes = self._run_task(
                    task,
                    task_offset=task_offset,
                    progress_bar=progress_bar,
                )
                task_results.append((task, episodes))
            return _metrics_from_task_results(task_results)
        finally:
            if last_result is not _MISSING:
                self.planner.last_result = last_result

    def close(self) -> None:
        for env in self._cached_envs.values():
            closer = getattr(env, "close", None)
            if callable(closer):
                closer()
        self._cached_envs.clear()

    def _run_task(
        self,
        task: EvalTask,
        *,
        task_offset: int,
        progress_bar: TqdmProgressBar | None,
    ) -> list[_EpisodeStats]:
        eval_env = self._eval_env_for_task(task)
        params = _materialized_params(self.learner, task)
        seed = self.seed + 17_000 + 97 * self._eval_count + int(task.task_index or 0)
        observations = np.asarray(eval_env.reset(seed=seed), dtype=np.float32)
        num_envs = int(eval_env.num_envs)
        returns = np.zeros((num_envs,), dtype=np.float64)
        lengths = np.zeros((num_envs,), dtype=np.int32)
        successes = np.zeros((num_envs,), dtype=np.float64)
        has_success = np.zeros((num_envs,), dtype=bool)
        finished = np.zeros((num_envs,), dtype=bool)
        task_label = _task_progress_label(task, task_offset, len(self.tasks))
        horizon = max(1, int(task.max_steps))

        for step_idx in range(horizon):
            if bool(np.all(finished)):
                break
            actions = _eval_actions(
                self.planner,
                observations,
                params=params,
                progress_bar=progress_bar,
                task_label=task_label,
                step_label=f"{step_idx + 1}/{horizon}",
            )
            observations, rewards, dones, infos = eval_env.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(num_envs)
            dones = np.asarray(dones, dtype=bool).reshape(num_envs)
            if len(infos) < num_envs:
                infos = list(infos) + [{} for _ in range(num_envs - len(infos))]
            for lane in range(num_envs):
                if finished[lane]:
                    continue
                returns[lane] += float(rewards[lane])
                lengths[lane] += 1
                info = infos[lane] if lane < len(infos) else {}
                if "success" in info:
                    has_success[lane] = True
                    successes[lane] = max(float(successes[lane]), float(info["success"]))
                if bool(dones[lane]):
                    finished[lane] = True

        if not bool(np.all(finished)) and progress_bar is not None:
            progress_bar.pulse(
                {
                    "phase": "eval",
                    "eval": task_label,
                    "eval_step": f"{horizon}/{horizon}",
                }
            )

        episodes: list[_EpisodeStats] = []
        for lane in range(num_envs):
            episodes.append(
                _EpisodeStats(
                    episode_return=float(returns[lane]),
                    episode_length=int(lengths[lane]),
                    success=float(successes[lane] >= 0.5) if bool(has_success[lane]) else None,
                )
            )
        if self._env_factory is None and task.kind != "gym":
            closer = getattr(eval_env, "close", None)
            if callable(closer):
                closer()
            self._cached_envs.pop(_cache_key(task), None)
        return episodes

    def _eval_env_for_task(self, task: EvalTask) -> EvalVectorEnv:
        if self._env_factory is not None:
            return self._env_factory(task)
        key = _cache_key(task)
        cached = self._cached_envs.get(key)
        if cached is not None:
            return cached
        env = _make_eval_env(task, num_envs=self.num_episodes, seed=self.seed)
        if task.kind == "gym":
            self._cached_envs[key] = env
        return env


@dataclass(frozen=True)
class _EpisodeStats:
    episode_return: float
    episode_length: int
    success: float | None


_MISSING = object()


def _cache_key(task: EvalTask) -> str:
    return f"{task.kind}:{task.benchmark}:{task.env_id}:{task.task_index}:{task.name}"


def _task_progress_label(task: EvalTask, task_offset: int, num_tasks: int) -> str:
    if num_tasks <= 1:
        return task.name
    return f"{task.name} {task_offset + 1}/{num_tasks}"


def _metrics_from_task_results(
    task_results: list[tuple[EvalTask, list[_EpisodeStats]]],
) -> dict[str, float]:
    task_returns: list[float] = []
    task_successes: list[float] = []
    metrics: dict[str, float] = {}
    for task, episodes in task_results:
        returns = [float(episode.episode_return) for episode in episodes]
        lengths = [float(episode.episode_length) for episode in episodes]
        mean_return = float(sum(returns) / max(1, len(returns)))
        mean_length = float(sum(lengths) / max(1, len(lengths)))
        task_returns.append(mean_return)
        safe_name = sanitize_task_name(task.name)
        metrics[f"eval/task/{safe_name}/mean_return"] = mean_return
        metrics[f"eval/task/{safe_name}/mean_length"] = mean_length
        successes = [float(episode.success) for episode in episodes if episode.success is not None]
        if successes:
            success_rate = float(sum(successes) / len(successes))
            task_successes.append(success_rate)
            metrics[f"eval/task/{safe_name}/success"] = success_rate
    metrics["eval/mean_return"] = float(sum(task_returns) / max(1, len(task_returns)))
    if task_successes:
        metrics["eval/mean_success"] = float(sum(task_successes) / len(task_successes))
    return metrics


def _materialized_params(learner: Learner, task: EvalTask) -> Any | None:
    materialize = getattr(learner, "materialize_task", None)
    if not callable(materialize):
        return None
    task_id = task.task_index
    if task_id is None:
        current = getattr(learner, "task_id", None)
        task_id = None if current is None else int(current)
    if task_id is None:
        return None
    return materialize(int(task_id))


def _eval_actions(
    planner: Planner,
    observations: np.ndarray,
    *,
    params: Any | None,
    progress_bar: TqdmProgressBar | None,
    task_label: str,
    step_label: str,
) -> np.ndarray:
    if progress_bar is not None:
        progress_bar.pulse(
            {
                "phase": "eval",
                "eval": task_label,
                "eval_step": step_label,
            }
        )
    obs_array = np.asarray(observations, dtype=np.float32)
    batch_size = int(obs_array.shape[0])
    if isinstance(planner, BatchedPlanner):
        obs_batch = [obs_array[lane] for lane in range(batch_size)]
        kwargs: dict[str, Any] = {"deterministic": True, "use_self_play": False}
        if params is not None:
            kwargs["params"] = params
        try:
            result = planner.search_batch(obs_batch, **kwargs)
        except TypeError:
            result = planner.search_batch(obs_batch, deterministic=True)
        return np.asarray(result.actions)
    actions = [
        np.asarray(planner.search(obs_array[lane], deterministic=True), dtype=np.float32)
        for lane in range(batch_size)
    ]
    return np.stack(actions, axis=0)


def _make_eval_env(task: EvalTask, *, num_envs: int, seed: int) -> EvalVectorEnv:
    if task.kind == "gym":
        if not task.env_id:
            raise TypeError("Gymnasium eval task is missing env_id.")
        return _GymnasiumEvalEnv(task.env_id, num_envs=num_envs, seed=seed)
    if task.kind == "cw":
        if not task.benchmark or task.task_index is None:
            raise TypeError("CW eval task is missing benchmark or task_index.")
        return _MtcworldEvalEnv.from_cw_task(
            task.benchmark,
            int(task.task_index),
            num_envs=num_envs,
            seed=seed,
            config=task.cw_config,
        )
    if task.kind == "mtcworld":
        if not task.env_id:
            raise TypeError("MTCWorld eval task is missing env_id.")
        return _MtcworldEvalEnv.from_env_id(
            task.env_id,
            num_envs=num_envs,
            seed=seed,
            config=task.cw_config,
        )
    raise TypeError(f"Unsupported eval task kind {task.kind!r}.")


class _GymnasiumEvalEnv:
    """``num_envs`` parallel Gymnasium episodes for evaluation."""

    def __init__(self, env_id: str, *, num_envs: int, seed: int) -> None:
        import gymnasium as gym

        from algorl.backends.jax.envs.gymnasium_vector import make_gymnasium_vector_env

        self._adapter = make_gymnasium_vector_env(
            lambda: gym.make(env_id),
            num_envs,
            seed=seed,
        )
        self.num_envs = int(num_envs)
        self._action_space = self._adapter.vector_env.single_action_space

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        observation, _ = self._adapter.vector_env.reset(seed=seed)
        return np.asarray(observation, dtype=np.float32)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        from algorl.backends.jax.envs.gymnasium_vector import _lane_infos_from_vector_infos

        vector_actions = _gym_vector_actions(
            actions,
            action_space=self._action_space,
            num_envs=self.num_envs,
        )
        next_observation, reward, terminations, truncations, infos = self._adapter.vector_env.step(
            vector_actions
        )
        done = np.asarray(terminations | truncations, dtype=bool)
        lane_infos = _lane_infos_from_vector_infos(infos, self.num_envs)
        return (
            np.asarray(next_observation, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            done,
            lane_infos,
        )

    def close(self) -> None:
        closer = getattr(self._adapter.vector_env, "close", None)
        if callable(closer):
            closer()


class _MtcworldEvalEnv:
    """Parallel CW / Sawyer eval lanes backed by an MTCWorld vector adapter."""

    def __init__(self, adapter: Any, *, seed: int) -> None:
        import jax

        self._adapter = adapter
        self.num_envs = int(adapter.num_envs)
        self._state = None
        self._key = jax.random.PRNGKey(int(seed))

    @classmethod
    def from_cw_task(
        cls,
        benchmark: str,
        task_index: int,
        *,
        num_envs: int,
        seed: int,
        config: Any = None,
    ) -> _MtcworldEvalEnv:
        from algorl.backends.jax.envs.mtcworld_jax import MtcworldCWRolloutCollector

        collector = MtcworldCWRolloutCollector(
            benchmark,
            task_index,
            num_envs=num_envs,
            seed=seed + task_index,
            config=config,
        )
        return cls(collector.jax_env, seed=seed)

    @classmethod
    def from_env_id(
        cls,
        env_id: str,
        *,
        num_envs: int,
        seed: int,
        config: Any = None,
    ) -> _MtcworldEvalEnv:
        from algorl.backends.jax.envs.mtcworld_jax import MtcworldRolloutCollector

        collector = MtcworldRolloutCollector(
            env_id,
            num_envs=num_envs,
            seed=seed,
            config=config,
        )
        return cls(collector.jax_env, seed=seed)

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        import jax

        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed))
        self._key, reset_key = jax.random.split(self._key)
        self._state = self._adapter.reset(reset_key)
        return np.asarray(self._adapter.observation(self._state), dtype=np.float32)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        import jax.numpy as jnp

        if self._state is None:
            raise RuntimeError("Eval env step() called before reset().")
        action_array = jnp.asarray(actions, dtype=jnp.float32)
        self._state = self._adapter.step(self._state, action_array)
        observation = np.asarray(self._adapter.observation(self._state), dtype=np.float32)
        reward = np.asarray(self._state.reward, dtype=np.float32)
        done = np.asarray(
            (self._state.truncated + self._state.terminated) > 0,
            dtype=bool,
        )
        success = None
        metrics = getattr(self._state, "metrics", None)
        if isinstance(metrics, dict):
            success = metrics.get("success")
        infos: list[dict[str, Any]] = []
        for lane in range(self.num_envs):
            info: dict[str, Any] = {}
            if success is not None:
                info["success"] = float(np.asarray(success).reshape(-1)[lane])
            infos.append(info)
        return observation, reward, done, infos

    def close(self) -> None:
        vector_env = getattr(self._adapter, "vector_env", None)
        closer = getattr(vector_env, "close", None)
        if callable(closer):
            closer()


def _gym_vector_actions(actions: np.ndarray, *, action_space: Any, num_envs: int) -> np.ndarray:
    import gymnasium as gym

    array = np.asarray(actions)
    if isinstance(action_space, gym.spaces.Discrete):
        return array.astype(np.int64).reshape(num_envs)
    return array.astype(np.float32).reshape((num_envs, *tuple(int(dim) for dim in action_space.shape)))


def _env_objects(train_env: TrainingEnv) -> list[Any]:
    seen: set[int] = set()
    ordered: list[Any] = []
    stack: list[Any] = [train_env, getattr(train_env, "raw", None), getattr(train_env, "unwrapped", None)]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        ordered.append(obj)
        for attr in ("raw", "unwrapped", "jax_env", "_adapter", "vector_env", "_vector_env"):
            child = getattr(obj, attr, None)
            if child is not None:
                stack.append(child)
        envs = getattr(obj, "envs", None)
        if isinstance(envs, (list, tuple)):
            stack.extend(envs)
    return ordered


def _gym_env_id(train_env: TrainingEnv) -> str | None:
    for obj in _env_objects(train_env):
        spec = getattr(obj, "spec", None)
        env_id = getattr(spec, "id", None) if spec is not None else None
        if isinstance(env_id, str) and env_id:
            return env_id
    return None


def _gym_max_episode_steps(train_env: TrainingEnv) -> int | None:
    for obj in _env_objects(train_env):
        spec = getattr(obj, "spec", None)
        value = getattr(spec, "max_episode_steps", None) if spec is not None else None
        if value:
            return int(value)
        wrapped = getattr(obj, "_max_episode_steps", None)
        if wrapped:
            return int(wrapped)
    return None


def _mtcworld_single_task_id(train_env: TrainingEnv) -> str | None:
    for obj in _env_objects(train_env):
        if getattr(obj, "benchmark_name", None) in CW_BENCHMARK_NAMES:
            return None
        for attr in ("env_id", "env_name"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value and value not in CW_BENCHMARK_NAMES:
                return value
    return None
