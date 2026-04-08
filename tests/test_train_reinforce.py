"""Tests for the current train_reinforce placeholder entrypoint."""

from __future__ import annotations

import pytest

from scripts import train_reinforce


def test_train_reinforce_exits_with_rollout_message() -> None:
    """The training entrypoint waits explicitly until learning is implemented."""
    with pytest.raises(SystemExit) as error:
        train_reinforce.main()

    assert (
        str(error.value)
        == "REINFORCE training is not implemented yet. "
        "Use rl_counterpoint.algos.rollout.collect_episode first."
    )
