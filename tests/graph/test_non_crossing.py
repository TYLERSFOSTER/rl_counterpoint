"""Tests for edge predicates in G(n)_1.

These tests cover transition-level rules after source and target nodes already
belong to G(n)_0. They verify voice crossing, parallel fifth detection across
all voice pairs, self-loop rejection, single-line upward interval limits, and
the boolean graph-spec knobs that allow or reject crossing/parallel-fifth edges.
"""

from __future__ import annotations

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.non_crossing import (
    has_parallel_fifth,
    has_voice_crossing,
    is_non_crossing,
    is_valid_edge,
    respects_single_line_interval,
)


def test_non_crossing_transition() -> None:
    """A simple parallel upward move does not cross adjacent voices."""
    assert not has_voice_crossing((3, 6), (4, 7))
    assert is_non_crossing((3, 6), (4, 7))


def test_crossing_from_below_is_detected() -> None:
    """A lower target voice reaching the prior upper voice is crossing."""
    assert has_voice_crossing((3, 6), (6, 9))


def test_crossing_from_above_is_detected() -> None:
    """An upper source voice reaching the new lower voice is crossing."""
    assert has_voice_crossing((3, 6), (0, 3))


def test_parallel_fifth_detected_for_any_voice_pair() -> None:
    """Parallel fifth detection ranges over all voice pairs, not just neighbors."""
    source = (3, 6, 10)
    target = (4, 8, 11)

    assert has_parallel_fifth(source, target)


def test_self_loop_is_invalid_edge() -> None:
    """Self-loops are excluded from G(n)_1."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_edge((3, 6), (3, 6), spec)


def test_invalid_source_or_target_is_invalid_edge() -> None:
    """An edge is invalid if either endpoint is outside G(n)_0."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_edge((0, 3), (3, 6), spec)
    assert not is_valid_edge((3, 6), (0, 3), spec)


def test_parallel_fifth_invalidates_edge_by_default() -> None:
    """Parallel fifths are rejected when the spec disallows them."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_edge((3, 10), (4, 11), spec)


def test_parallel_fifth_can_be_allowed() -> None:
    """Parallel fifths are accepted when the spec explicitly allows them."""
    spec = CounterpointGraphSpec(n=2, tonic=60, allow_parallel_fifths=True)

    assert is_valid_edge((3, 10), (4, 11), spec)


def test_voice_crossing_invalidates_edge_by_default() -> None:
    """Voice crossing is rejected when the spec disallows it."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_edge((3, 6), (6, 9), spec)


def test_voice_crossing_can_be_allowed() -> None:
    """Voice crossing is accepted when the spec explicitly allows it."""
    spec = CounterpointGraphSpec(n=2, tonic=60, allow_voice_crossing=True)

    assert is_valid_edge((3, 6), (7, 10), spec)


def test_single_line_upward_interval_above_max_invalidates_edge() -> None:
    """A voice moving upward by more than M violates the edge interval trim."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert respects_single_line_interval((3, 6), (14, 17), spec)
    assert not respects_single_line_interval((3, 6), (15, 18), spec)
    assert not is_valid_edge((3, 6), (15, 18), spec)


def test_large_downward_motion_is_currently_allowed_by_interval_rule() -> None:
    """The current edge interval trim caps upward motion, not absolute motion."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert respects_single_line_interval((20, 23), (3, 6), spec)
