"""Training script placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AlgoRL agent.")
    parser.add_argument("--agent", default="efficient_zero")
    parser.add_argument("--backend", default="jax")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Training for agent={args.agent!r} backend={args.backend!r} is not implemented yet."
    )


if __name__ == "__main__":
    main()
