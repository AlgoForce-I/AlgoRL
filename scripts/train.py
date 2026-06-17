"""Training script placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AlgoRL agent.")
    parser.add_argument("--agent", default="efficient_zero")
    parser.add_argument("--backend", default="jax")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Implement:
    # 1. env = gymnasium.make(args.env_id)
    # 2. agent = selected AlgoRL agent class, e.g. arl.EfficientZero(env, backend=args.backend)
    # 3. agent.learn(total_timesteps=args.total_timesteps)
    # 4. optionally save a checkpoint and close the environment
    raise NotImplementedError(
        f"Training for agent={args.agent!r} backend={args.backend!r} is not implemented yet."
    )


if __name__ == "__main__":
    main()
