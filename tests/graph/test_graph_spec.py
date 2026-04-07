"""Tests for the CounterpointGraphSpec contract.

These tests protect the object that defines which graph G(n) we mean. They
check that the core graph knobs construct correctly, that derived values like
pitch classes and chord width are computed as expected, and that invalid graph
definitions fail immediately instead of leaking into state/edge logic.
"""

from __future__ import annotations

import pytest

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec


def test_valid_spec_constructs() -> None:
    """A default valid spec exposes the expected graph parameters and defaults."""
    spec = CounterpointGraphSpec(n=3, tonic=60)

    assert spec.n == 3
    assert spec.tonic == 60
    assert spec.pitch_count == 128
    assert spec.tonic_pitch_class == 0
    assert spec.max_chord_width == 18
    assert spec.max_adjacent_vertical_interval == 11
    assert spec.max_single_line_interval == 11
    assert spec.allowed_root_pitch_classes == frozenset({3, 4, 5, 7, 8, 9})
    assert spec.allowed_outer_interval_classes == frozenset({3, 4, 5, 7, 8, 9})
    assert not spec.allow_voice_crossing
    assert not spec.allow_parallel_fifths


def test_tonic_pitch_class_wraps_mod_12() -> None:
    """The tonic and root pitch classes are computed modulo 12."""
    spec = CounterpointGraphSpec(n=2, tonic=61)

    assert spec.tonic_pitch_class == 1
    assert spec.allowed_root_pitch_classes == frozenset({4, 5, 6, 8, 9, 10})


def test_invalid_n_raises() -> None:
    """The graph must have at least one voice."""
    with pytest.raises(ValueError, match="n must be at least 1"):
        CounterpointGraphSpec(n=0, tonic=60)


def test_invalid_pitch_range_raises() -> None:
    """The pitch range lower bound cannot exceed the upper bound."""
    with pytest.raises(ValueError, match="pitch_min must be <= pitch_max"):
        CounterpointGraphSpec(n=2, tonic=60, pitch_min=10, pitch_max=9)


def test_pitch_range_smaller_than_n_raises() -> None:
    """The pitch range must contain enough distinct pitches for n voices."""
    with pytest.raises(ValueError, match="pitch range must contain at least n pitches"):
        CounterpointGraphSpec(n=3, tonic=60, pitch_min=0, pitch_max=1)


def test_invalid_max_interval_raises() -> None:
    """The shared max interval must be a positive integer."""
    with pytest.raises(ValueError, match="max_interval must be at least 1"):
        CounterpointGraphSpec(n=2, tonic=60, max_interval=0)


def test_invalid_interval_set_values_raise() -> None:
    """Mod-12 interval sets must contain only pitch-class values from 0 to 11."""
    with pytest.raises(ValueError, match="allowed_root_intervals_mod_12"):
        CounterpointGraphSpec(
            n=2,
            tonic=60,
            allowed_root_intervals_mod_12=frozenset({12}),
        )

    with pytest.raises(ValueError, match="allowed_outer_intervals_mod_12"):
        CounterpointGraphSpec(
            n=2,
            tonic=60,
            allowed_outer_intervals_mod_12=frozenset({-1}),
        )
