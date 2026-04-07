"""Edge predicates for the counterpoint graph G(n)."""

from __future__ import annotations

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.state_space import ChordState, is_valid_node


def has_voice_crossing(source: ChordState, target: ChordState) -> bool:
    """Return True iff source -> target crosses adjacent voice boundaries."""
    if len(source) != len(target):
        raise ValueError("source and target must have the same length")

    return any(
        target[i] >= source[i + 1] or source[i] >= target[i + 1]
        for i in range(len(source) - 1)
    )


def is_non_crossing(source: ChordState, target: ChordState) -> bool:
    return not has_voice_crossing(source, target)


def has_parallel_fifth(source: ChordState, target: ChordState) -> bool:
    """Return True iff any voice pair forms a parallel perfect fifth."""
    if len(source) != len(target):
        raise ValueError("source and target must have the same length")

    n = len(source)
    for i in range(n):
        displacement_i = target[i] - source[i]

        for j in range(i + 1, n):
            if source[j] - source[i] != 7:
                continue
            if target[j] - source[j] == displacement_i:
                return True

    return False


def respects_single_line_interval(
    source: ChordState,
    target: ChordState,
    spec: CounterpointGraphSpec,
) -> bool:
    """Return True iff every voice satisfies mu_i - lambda_i <= M."""
    if len(source) != len(target):
        raise ValueError("source and target must have the same length")

    return all(
        target_pitch - source_pitch <= spec.max_single_line_interval
        for source_pitch, target_pitch in zip(source, target, strict=True)
    )


def is_valid_edge(
    source: ChordState,
    target: ChordState,
    spec: CounterpointGraphSpec,
) -> bool:
    """Return True iff source -> target belongs to the trimmed edge set G(n)_1."""
    if source == target:
        return False

    if not is_valid_node(source, spec) or not is_valid_node(target, spec):
        return False

    if not respects_single_line_interval(source, target, spec):
        return False

    if not spec.allow_voice_crossing and has_voice_crossing(source, target):
        return False

    if not spec.allow_parallel_fifths and has_parallel_fifth(source, target):
        return False

    return True
