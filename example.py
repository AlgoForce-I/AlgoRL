"""Minimal AlgoRL usage example.

Run from the project root::

    python example.py

Install the package and JAX backend first::

    pip install -e ".[jax]"
"""

from __future__ import annotations

import algorl as arl
from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.envs import make_batched_cw_train_env
from algorl.envs import resolve_env
from MTCWorldMJX import CWConfig

NUM_ENVS = 32


def main() -> None:
    cw_config = CWConfig(seed=42, steps_per_task=1_000_000)
    jax_env = make_batched_cw_train_env(
        "CW10",
        num_envs=NUM_ENVS,
        seed=42,
        config=cw_config,
    )
    env = resolve_env(jax_env, seed=42)

    ez_config = EfficientZeroConfig.for_dmc_state_batched_cl(
        num_envs=NUM_ENVS,
        seed=0,
        buffer_capacity=10_000,
        learning_starts=1_000,
        reanalyze_ratio=1,
        gradient_steps_per_rollout=3
    )

    agent = arl.EfficientZero(env, config=ez_config)

    print(f"AlgoRL {arl.__version__}")
    print(
        f"Environment: CW10 batched continual "
        f"({jax_env.current_task_name} -> ... x {env.num_envs} envs/task)"
    )
    print(f"Observation shape: {env.observation_shape}")
    print(f"Steps per task: {jax_env.steps_per_task:,}")
    print(f"Backend: {agent.backend.name}")
    print(f"World model: {type(agent.world_model).__name__}")
    print(f"Planner: {type(agent.planner).__name__}")
    print(f"Learner: {type(agent.learner).__name__}")
    print(f"Replay buffer: {type(agent.replay_buffer).__name__}")
    print(f"Batched env: {env.is_batched}")
    print(f"JIT MCTS: {getattr(agent.planner, '_use_jit', False)}")
    print(f"Search batch size: {agent.config.search_batch_size}")
    print(f"Rollout chunk: {agent.config.jax_rollout_chunk}")
    print(f"Learner batch size: {agent.config.batch_size}")
    print(f"Reanalyze ratio: {agent.config.reanalyze_ratio}")
    print(f"Gradient steps per rollout: {agent.config.gradient_steps_per_rollout}")

    agent.learn(
        total_timesteps=10_000_000,
        tensorboard_log_dir="runs/cw10_ez_batched_cl",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
