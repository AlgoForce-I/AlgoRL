"""Uniform replay buffer tests."""

from algorl.buffers.replay import UniformReplayBuffer
from algorl.core.types import Transition


def test_uniform_replay_buffer_add_and_sample() -> None:
    buffer = UniformReplayBuffer(capacity=10)
    for index in range(5):
        buffer.add(
            Transition(
                observation=index,
                action=0,
                reward=1.0,
                next_observation=index + 1,
                done=False,
            )
        )

    batch = buffer.sample(3)
    assert len(batch.data["transitions"]) == 3
    assert len(buffer) == 5
