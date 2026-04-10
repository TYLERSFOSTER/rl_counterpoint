"""Shared interval helpers for the music layer."""

from __future__ import annotations


def pitch_class_interval(lower_pitch: int, upper_pitch: int) -> int:
    """Return the octave-equivalent pitch-class interval from lower to upper."""
    return (upper_pitch - lower_pitch) % 12
