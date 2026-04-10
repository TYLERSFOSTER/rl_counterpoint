"""Shared consonance helpers for the music layer."""

from __future__ import annotations

JUST_INTERVAL_RATIOS_BY_PITCH_CLASS: dict[int, tuple[int, int]] = {
    0: (1, 1),
    1: (16, 15),
    2: (9, 8),
    3: (6, 5),
    4: (5, 4),
    5: (4, 3),
    6: (45, 32),
    7: (3, 2),
    8: (8, 5),
    9: (5, 3),
    10: (9, 5),
    11: (15, 8),
}


def just_ratio_height(interval_pitch_class: int) -> int:
    """Return p + q for the chosen just-ratio prototype of one pitch class."""
    numerator, denominator = JUST_INTERVAL_RATIOS_BY_PITCH_CLASS[interval_pitch_class]
    return numerator + denominator


def consonance_from_pitch_class(interval_pitch_class: int) -> float:
    """Return the project's static consonance score for one pitch class."""
    return 1.0 / just_ratio_height(interval_pitch_class)
