"""Node/state-space helpers for the counterpoint graph G(n)."""

from __future__ import annotations

from itertools import product
from typing import TypeAlias

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec

ChordState: TypeAlias = tuple[int, ...]


def pitch_class(pitch: int) -> int:
    return pitch % 12


def adjacent_intervals(state: ChordState) -> tuple[int, ...]:
    return tuple(state[i + 1] - state[i] for i in range(len(state) - 1))


def outer_interval(state: ChordState) -> int:
    if not state:
        raise ValueError("state must not be empty")

    return state[-1] - state[0]


def is_strictly_increasing(state: ChordState) -> bool:
    return all(state[i] < state[i + 1] for i in range(len(state) - 1))


def is_in_pitch_range(state: ChordState, spec: CounterpointGraphSpec) -> bool:
    return all(spec.pitch_min <= pitch <= spec.pitch_max for pitch in state)


def has_valid_length(state: ChordState, spec: CounterpointGraphSpec) -> bool:
    return len(state) == spec.n


def has_valid_adjacent_intervals(
    state: ChordState,
    spec: CounterpointGraphSpec,
) -> bool:
    for interval in adjacent_intervals(state):
        if interval > spec.max_adjacent_vertical_interval:
            return False
        if interval in spec.forbidden_adjacent_vertical_intervals:
            return False

    return True


def has_valid_outer_interval(state: ChordState, spec: CounterpointGraphSpec) -> bool:
    interval = outer_interval(state)
    if interval > spec.max_chord_width:
        return False

    return interval % 12 in spec.allowed_outer_interval_classes


def has_valid_root(state: ChordState, spec: CounterpointGraphSpec) -> bool:
    if not state:
        return False

    return pitch_class(state[0]) in spec.allowed_root_pitch_classes


def is_valid_node(state: ChordState, spec: CounterpointGraphSpec) -> bool:
    """Return True iff state belongs to the trimmed node set G(n)_0."""
    return (
        has_valid_length(state, spec)
        and is_in_pitch_range(state, spec)
        and is_strictly_increasing(state)
        and has_valid_adjacent_intervals(state, spec)
        and has_valid_outer_interval(state, spec)
        and has_valid_root(state, spec)
    )


def iter_gap_vectors(spec: CounterpointGraphSpec) -> tuple[tuple[int, ...], ...]:
    """Return admissible gap vectors for the trimmed node set G(n)_0."""
    if spec.n == 1:
        return ((),)

    max_gap = min(spec.max_adjacent_vertical_interval, spec.max_chord_width)
    allowed_gaps = tuple(
        gap
        for gap in range(1, max_gap + 1)
        if gap not in spec.forbidden_adjacent_vertical_intervals
    )
    gap_vectors = []

    for candidate in product(allowed_gaps, repeat=spec.n - 1):
        candidate_width = sum(candidate)
        if (
            candidate_width <= spec.max_chord_width
            and candidate_width % 12 in spec.allowed_outer_interval_classes
        ):
            gap_vectors.append(candidate)

    return tuple(gap_vectors)


def state_from_root_and_gaps(root: int, gaps: tuple[int, ...]) -> ChordState:
    state = [root]
    current = root

    for gap in gaps:
        current += gap
        state.append(current)

    return tuple(state)


def iter_node_states(spec: CounterpointGraphSpec) -> tuple[ChordState, ...]:
    """Return all node states in G(n)_0 for the current spec.

    This is intended for small, trimmed research graphs and smoke checks. Later
    environment code may prefer lazy generators or action-local candidates.
    """
    states = []

    for gaps in iter_gap_vectors(spec):
        width = sum(gaps)
        for root in range(spec.pitch_min, spec.pitch_max - width + 1):
            state = state_from_root_and_gaps(root, gaps)
            if has_valid_root(state, spec):
                states.append(state)

    return tuple(states)
