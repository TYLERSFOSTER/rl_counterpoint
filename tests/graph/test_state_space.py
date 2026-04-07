"""Tests for node membership in G(n)_0.

These tests cover the trimmed chord/node set before edge logic is considered.
They verify length, MIDI pitch range, strict voice ordering, adjacent vertical
interval rules, total chord width, root scale membership, outer interval
consonance, and helper generation of valid gap vectors and node states.
"""

from __future__ import annotations

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.state_space import (
    adjacent_intervals,
    is_valid_node,
    iter_gap_vectors,
    iter_node_states,
    outer_interval,
    state_from_root_and_gaps,
)


def test_valid_node_passes() -> None:
    """A tuple satisfying all G(n)_0 trims is accepted as a valid node."""
    spec = CounterpointGraphSpec(n=3, tonic=60)

    assert is_valid_node((3, 6, 10), spec)


def test_wrong_length_fails() -> None:
    """A node candidate must have exactly n voices."""
    spec = CounterpointGraphSpec(n=3, tonic=60)

    assert not is_valid_node((3, 6), spec)


def test_out_of_range_pitch_fails() -> None:
    """A node candidate cannot use pitches outside the MIDI pitch range."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_node((3, 128), spec)


def test_non_strict_or_duplicate_pitch_fails() -> None:
    """A node candidate must be strictly increasing from low to high voice."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_node((3, 3), spec)
    assert not is_valid_node((6, 3), spec)


def test_adjacent_interval_above_max_fails() -> None:
    """Adjacent vertical intervals cannot exceed the graph max interval."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_node((3, 15), spec)


def test_forbidden_adjacent_interval_fails() -> None:
    """Adjacent dissonant intervals are rejected by the node trim."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    for interval in (1, 2, 6, 10, 11):
        assert not is_valid_node((3, 3 + interval), spec)


def test_total_chord_width_above_cap_fails() -> None:
    """The outer chord width cannot exceed ceil(n * width_factor)."""
    spec = CounterpointGraphSpec(n=3, tonic=60)

    assert not is_valid_node((3, 12, 24), spec)


def test_outer_interval_mod_12_must_be_allowed() -> None:
    """The outer interval must be consonant modulo 12."""
    spec = CounterpointGraphSpec(n=3, tonic=60)

    assert not is_valid_node((3, 7, 13), spec)


def test_root_pitch_class_must_match_tonic_rule() -> None:
    """The root pitch class must satisfy the tonic-relative scale trim."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert not is_valid_node((0, 3), spec)


def test_interval_helpers() -> None:
    """Interval helpers report adjacent intervals and total outer width."""
    state = (3, 6, 15)

    assert adjacent_intervals(state) == (3, 9)
    assert outer_interval(state) == 12


def test_state_from_root_and_gaps() -> None:
    """A root and gap vector reconstruct the corresponding chord state."""
    assert state_from_root_and_gaps(3, (4, 5)) == (3, 7, 12)


def test_iter_gap_vectors_returns_only_valid_gap_patterns() -> None:
    """Generated gap vectors satisfy all gap-level node trims."""
    spec = CounterpointGraphSpec(n=3, tonic=60)

    for gaps in iter_gap_vectors(spec):
        assert all(gap <= spec.max_adjacent_vertical_interval for gap in gaps)
        assert all(gap not in spec.forbidden_adjacent_vertical_intervals for gap in gaps)
        assert sum(gaps) <= spec.max_chord_width
        assert sum(gaps) % 12 in spec.allowed_outer_interval_classes


def test_iter_node_states_returns_only_valid_nodes() -> None:
    """Every generated node state satisfies the full node predicate."""
    spec = CounterpointGraphSpec(n=3, tonic=60)

    assert all(is_valid_node(state, spec) for state in iter_node_states(spec))


def test_iter_node_states_matches_known_n2_count() -> None:
    """The generated two-voice node count matches the formula-script count."""
    spec = CounterpointGraphSpec(n=2, tonic=60)

    assert len(iter_node_states(spec)) == 363
