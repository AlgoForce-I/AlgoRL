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
TOTAL_ENV_STEPS = 10_000_000


def main() -> None:
    gym_env = GymnasiumSearchEnv(gym.make("HalfCheetah-v5"))
    env = resolve_env(gym_env, seed=42)

    # schedule_horizon="auto" (default): temperature, priority beta, and mixed-value
    # milestones scale with learn(total_timesteps=...) using HyperCEZ alt2 ratios.
    ez_config = EfficientZeroConfig.for_dmc_state_sequential_gpu(
        batch_size=256,
        reanalyze_ratio=1.0,
        reanalyze_search_batch_size=REANALYZE_SEARCH_BATCH_SIZE,
        seed=42,
        train_freq=1,
        learning_starts=2_000,
    )

    agent = arl.EfficientZero(env, config=ez_config)
    resolved = ez_config.with_schedule_for_run(TOTAL_ENV_STEPS)

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
    print(f"Schedule horizon: {agent.config.schedule_horizon}")
    print(f"Auto gradient budget (for {TOTAL_ENV_STEPS:,} env steps): {resolved.total_training_steps:,}")
    print(f"Mix training starts at grad step: {resolved.start_use_mix_training_steps:,}")
    print(f"Auto TD steps horizon: {resolved.auto_td_steps:,}")
    print(f"Mixed value replay window: {resolved.mixed_value_threshold:,.0f}")

    agent.learn(
        total_timesteps=TOTAL_ENV_STEPS,
        tensorboard_log_dir="runs/half_cheetah_ez",
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
