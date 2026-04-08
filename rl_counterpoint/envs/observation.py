"""Observation helpers for counterpoint environments."""

from __future__ import annotations

from dataclasses import dataclass

from rl_counterpoint.graph.state_space import ChordState

PAD_METRICAL_POSITION = -1


def build_observation(state: ChordState) -> ChordState:
    """Return the first environment observation representation."""
    return state


@dataclass(frozen=True)
class TimedChordWindow:
    """Fixed-length sequence observation built from raw env history."""

    chord_sequence: tuple[ChordState, ...]
    bar_positions: tuple[int, ...]
    valid_mask: tuple[bool, ...]


def pad_chord(*, chord_size: int) -> ChordState:
    """Return the distinguished PAD chord for sequence observations."""
    if chord_size < 1:
        raise ValueError("chord_size must be at least 1")

    return (0,) * chord_size


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


def build_timed_chord_window(
    *,
    history: tuple[ChordState, ...],
    step_index: int,
    measure_size: int,
    context_measures: int = 3,
) -> TimedChordWindow:
    """Build a fixed-length left-padded sequence window from raw env history."""
    if not history:
        raise ValueError("history must not be empty")
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")
    if context_measures < 1:
        raise ValueError("context_measures must be at least 1")

    chord_size = len(history[0])
    if chord_size < 1:
        raise ValueError("history chord states must not be empty")
    if any(len(chord) != chord_size for chord in history):
        raise ValueError("all chord states in history must have the same length")

    window_length = context_measures * measure_size
    real_history = history[-window_length:]
    real_start_step = step_index - len(real_history) + 1
    padding_length = window_length - len(real_history)
    pad_state = pad_chord(chord_size=chord_size)

    padded_chords = (pad_state,) * padding_length + real_history
    padded_bar_positions = (PAD_METRICAL_POSITION,) * padding_length + tuple(
        bar_position(step_index=real_start_step + offset, measure_size=measure_size)
        for offset in range(len(real_history))
    )
    valid_mask = (False,) * padding_length + (True,) * len(real_history)

    return TimedChordWindow(
        chord_sequence=padded_chords,
        bar_positions=padded_bar_positions,
        valid_mask=valid_mask,
    )
