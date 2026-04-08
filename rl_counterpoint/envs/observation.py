"""Observation helpers for counterpoint environments."""

from __future__ import annotations

from rl_counterpoint.graph.state_space import ChordState


def build_observation(state: ChordState) -> ChordState:
    """Return the first environment observation representation."""
    return state


def bar_position(*, step_index: int, measure_size: int) -> int:
    """Return the quarter-note position within the current m/4 bar."""
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")

    return step_index % measure_size


def is_leading_beat(*, step_index: int, measure_size: int) -> bool:
    """Return True iff the current step is the leading beat of the bar."""
    return bar_position(step_index=step_index, measure_size=measure_size) == 0


def is_downbeat(*, step_index: int) -> bool:
    """Return True iff the current quarter-note step is even-indexed."""
    return step_index % 2 == 0


def is_ending_beat(*, step_index: int, measure_size: int) -> bool:
    """Return True iff the current step is the final beat position of the bar."""
    return bar_position(step_index=step_index, measure_size=measure_size) == measure_size - 1
