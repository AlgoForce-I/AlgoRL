"""Throughput benchmark — batched HalfCheetah (parallel Gymnasium envs).

One point of a scaling sweep. The sweep varies parallelism only; everything
else, including the replay ratio, is held fixed, so throughput stays
comparable between points and between implementations::

    for n in 1 2 4 8 16 32; do
        BENCH_NUM_ENVS=$n BENCH_STEPS=100000 python example2_batched.py
    done

Knobs (all optional)::

    BENCH_NUM_ENVS   parallel env lanes (default 32)
    BENCH_SIMS       MCTS simulations per move (default: config preset)
    BENCH_STEPS      env steps to run (default 100_000)
    BENCH_SEED       seed (default 0)
    BENCH_TAG        extra label in the run directory name
    BENCH_OVERRIDES  JSON dict of extra config overrides

Each run writes TensorBoard scalars under ``bench/*`` plus a ``benchmark.json``
holding the config, the hardware and the final summary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from algorl.backends.jax.memory import configure_jax_gpu_memory

configure_jax_gpu_memory(preallocate=False, memory_fraction=0.85)

import gymnasium as gym

import algorl as arl
from algorl.common.benchmark import ThroughputCallback, write_benchmark_record

ENV_ID = "HalfCheetah-v5"
NUM_ENVS = int(os.environ.get("BENCH_NUM_ENVS", "32"))
TOTAL_TIMESTEPS = int(os.environ.get("BENCH_STEPS", "100000"))
SEED = int(os.environ.get("BENCH_SEED", "0"))
SIMULATIONS = os.environ.get("BENCH_SIMS")
LEARNING_STARTS = os.environ.get("BENCH_LEARNING_STARTS")
TAG = os.environ.get("BENCH_TAG", "")
# Arbitrary config overrides as JSON, for matching another implementation's
# hyperparameters exactly: BENCH_OVERRIDES='{"td_steps": 5, "batch_size": 256}'
OVERRIDES = json.loads(os.environ.get("BENCH_OVERRIDES", "{}"))

LR_DECAY_STEPS = 2_000_000
LR_DECAY_RATE = 0.1
RUN_NAME = f"hc_e{NUM_ENVS}{'_' + TAG if TAG else ''}_seed{SEED}"
TENSORBOARD_LOG_DIR = f"runs/bench/{RUN_NAME}"


def for_half_cheetah_batched(
    num_envs: int = 32,
    **overrides: object,
) -> arl.EfficientZeroConfig:
    """Batched Gym HalfCheetah preset (wider value support than DMC default).

    ``for_batched`` derives the rollout chunk and the gradient burst from
    ``num_envs`` together, so the replay ratio stays at one gradient step per
    env step at every point of the sweep. The benchmark logs the realised ratio
    (``bench/grad_steps_per_env_step``) rather than trusting it.
    """
    config = arl.EfficientZeroConfig.for_batched(num_envs=num_envs).with_overrides(
        value_support_range=(-5000.0, 5000.0),
        clip_inference_values=False,
        reward_support_range=(-10.0, 10.0),
        use_bn=True,
        lr_warm_up=0.01,
    ).with_schedule_for_run(
        TOTAL_TIMESTEPS,
        num_envs=num_envs,
    )

    if SIMULATIONS is not None:
        overrides.setdefault("mcts_simulations", int(SIMULATIONS))
    if LEARNING_STARTS is not None:
        overrides.setdefault("learning_starts", int(LEARNING_STARTS))
    for key, value in OVERRIDES.items():
        overrides.setdefault(key, value)

    return config.with_overrides(
        schedule_horizon="fixed",
        total_training_steps=config.total_training_steps,
        start_use_mix_training_steps=config.start_use_mix_training_steps,
        auto_td_steps=config.auto_td_steps,
        mixed_value_threshold=config.mixed_value_threshold,
        lr_decay_steps=LR_DECAY_STEPS,
        lr_decay_rate=LR_DECAY_RATE,
        seed=SEED,
        **overrides,
    )


def main() -> None:
    config = for_half_cheetah_batched(num_envs=NUM_ENVS)
    # The agent vectorises this to config.rollout_envs lanes and raises if the
    # count disagrees with the config, so the benchmark records what it ran.
    agent = arl.EfficientZero(gym.make(ENV_ID), config=config)
    throughput = ThroughputCallback(agent.learner, log_every_steps=100)

    record = write_benchmark_record(
        TENSORBOARD_LOG_DIR,
        name=RUN_NAME,
        config=config,
        extra={
            "env_id": ENV_ID,
            "num_envs": agent.env.num_envs,
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "planned_grad_steps_per_env_step": (
                config.gradient_steps_per_rollout / (NUM_ENVS * config.jax_rollout_chunk)
            ),
        },
    )
    print(f"[bench] {RUN_NAME}: {NUM_ENVS} envs, {config.mcts_simulations} sims, "
          f"{TOTAL_TIMESTEPS:,} steps -> {TENSORBOARD_LOG_DIR}")

    try:
        agent.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tensorboard_log_dir=TENSORBOARD_LOG_DIR,
            progress_bar=True,
            callbacks=throughput,
        )
    finally:
        summary = throughput.summary()
        payload = json.loads(Path(record).read_text())
        payload["summary"] = summary
        Path(record).write_text(json.dumps(payload, indent=2, sort_keys=True))
        def _fmt(key: str, unit: str = "") -> str:
            value = summary.get(key)
            return f"{value:.2f}{unit}" if isinstance(value, (int, float)) else "n/a"

        print(f"[bench] {RUN_NAME}")
        print(f"[bench]   env steps/s (steady) : {_fmt('bench/env_steps_per_s_steady')}")
        print(f"[bench]   env steps/s (all)    : {_fmt('bench/env_steps_per_s_avg')}")
        print(f"[bench]   grad steps/env step  : {_fmt('bench/grad_steps_per_env_step')}"
              "   <- must match across the sweep")
        print(f"[bench]   train time fraction  : {_fmt('bench/train_time_frac')}")
        print(f"[bench]   gpu utilisation      : {_fmt('bench/gpu_util_pct', '%')}")
        print(f"[bench]   startup (incl. JIT)  : {_fmt('startup_s', 's')}")
        print(f"[bench]   summary written to {record}")


if __name__ == "__main__":
    main()
