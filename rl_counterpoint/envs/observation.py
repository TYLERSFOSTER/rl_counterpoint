"""Observation helpers for counterpoint environments."""

from __future__ import annotations

from rl_counterpoint.graph.state_space import ChordState


def build_observation(state: ChordState) -> ChordState:
    """Return the first environment observation representation."""
    return state
