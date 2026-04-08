"""Tests for timed sequence observation helpers."""

from __future__ import annotations

import pytest

from rl_counterpoint.envs.observation import (
    PAD_METRICAL_POSITION,
    TimedChordWindow,
    build_timed_chord_window,
    pad_chord,
)


def test_pad_chord_returns_zero_pad_of_requested_size() -> None:
    """Sequence padding uses a distinguished zero chord of fixed size."""
    assert pad_chord(chord_size=3) == (0, 0, 0)


def test_build_timed_chord_window_left_pads_short_history() -> None:
    """Short histories are left-padded to a fixed three-measure window."""
    window = build_timed_chord_window(
        history=((3, 6),),
        step_index=0,
        measure_size=4,
        context_measures=3,
    )

    assert isinstance(window, TimedChordWindow)
    assert len(window.chord_sequence) == 12
    assert len(window.bar_positions) == 12
    assert len(window.valid_mask) == 12
    assert window.chord_sequence[-1] == (3, 6)
    assert window.bar_positions[-1] == 0
    assert window.valid_mask[-1]
    assert all(chord == (0, 0) for chord in window.chord_sequence[:-1])
    assert all(position == PAD_METRICAL_POSITION for position in window.bar_positions[:-1])
    assert all(not valid for valid in window.valid_mask[:-1])


def test_build_timed_chord_window_truncates_to_last_context_window() -> None:
    """Long histories are truncated to the most recent fixed-length suffix."""
    history = tuple((pitch, pitch + 3) for pitch in range(20))

    window = build_timed_chord_window(
        history=history,
        step_index=19,
        measure_size=4,
        context_measures=3,
    )

    assert len(window.chord_sequence) == 12
    assert window.chord_sequence[0] == history[-12]
    assert window.chord_sequence[-1] == history[-1]
    assert window.valid_mask == (True,) * 12
    assert window.bar_positions == (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3)


def test_build_timed_chord_window_rejects_invalid_inputs() -> None:
    """The window builder validates non-empty history and positive meter/window."""
    with pytest.raises(ValueError, match="history must not be empty"):
        build_timed_chord_window(
            history=(),
            step_index=0,
            measure_size=4,
        )

    with pytest.raises(ValueError, match="measure_size must be at least 1"):
        build_timed_chord_window(
            history=((3, 6),),
            step_index=0,
            measure_size=0,
        )

    with pytest.raises(ValueError, match="context_measures must be at least 1"):
        build_timed_chord_window(
            history=((3, 6),),
            step_index=0,
            measure_size=4,
            context_measures=0,
        )

    with pytest.raises(ValueError, match="all chord states in history must have the same length"):
        build_timed_chord_window(
            history=((3, 6), (4, 7, 9)),
            step_index=1,
            measure_size=4,
        )
