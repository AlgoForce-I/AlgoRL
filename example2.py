"""Minimal AlgoRL usage example — HalfCheetah (single Gymnasium env).

Run from the project root::

    python example2.py

Install the package and JAX backend first::

    pip install -e ".[jax]"
"""

from __future__ import annotations

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import algorl as arl
from algorl.agents.configs import EfficientZeroConfig
from algorl.envs import resolve_env

import gymnasium as gym
from algorl.backends.jax.envs import GymnasiumSearchEnv

NUM_ENVS = 1  # sequential gym env, not batched CW
# Reanalyze runs ``reanalyze_search_batch_size`` MCTS roots per JIT call.
# Rollout MCTS stays width 1; increase this (64–256) if reanalyze is slow.
REANALYZE_SEARCH_BATCH_SIZE = 2048


def main() -> None:
    gym_env = GymnasiumSearchEnv(gym.make("HalfCheetah-v5"))
    env = resolve_env(gym_env, seed=42)

    ez_config = EfficientZeroConfig.for_dmc_state_sequential_gpu(
        batch_size=256,
        reanalyze_ratio=1.0,
        reanalyze_search_batch_size=REANALYZE_SEARCH_BATCH_SIZE,
        seed=0,
        buffer_capacity=100_000,
        learning_starts=2_000,
        total_training_steps=100_000,
    )

    agent = arl.EfficientZero(env, config=ez_config)

    print(f"AlgoRL {arl.__version__}")
    print(f"Observation shape: {env.observation_shape}")
    print(f"Backend: {agent.backend.name}")
    print(f"World model: {type(agent.world_model).__name__}")
    print(f"Planner: {type(agent.planner).__name__}")
    print(f"Learner: {type(agent.learner).__name__}")
    print(f"Replay buffer: {type(agent.replay_buffer).__name__}")
    print(f"Batched env: {env.is_batched}")
    print(f"JIT MCTS: {getattr(agent.planner, '_use_jit', False)}")
    print(f"Rollout search batch size: {agent.config.search_batch_size}")
    print(f"Reanalyze search batch size: {agent.config.reanalyze_search_batch_size}")
    print(f"Rollout chunk: {agent.config.jax_rollout_chunk}")
    print(f"Learner batch size: {agent.config.batch_size}")
    print(f"Reanalyze ratio: {agent.config.reanalyze_ratio}")
    print(f"Gradient steps per rollout: {agent.config.gradient_steps_per_rollout}")

    agent.learn(
        total_timesteps=10_000_000,
        tensorboard_log_dir="runs/half_cheetah_ez",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
