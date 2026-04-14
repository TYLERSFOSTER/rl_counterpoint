"""Graph-level parameters defining the counterpoint graph G(n)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

DEFAULT_FORBIDDEN_ADJACENT_VERTICAL_INTERVALS = frozenset({1, 2, 6, 10, 11})
DEFAULT_ALLOWED_ROOT_INTERVALS_MOD_12 = frozenset({3, 4, 5, 7, 8, 9})
DEFAULT_ALLOWED_OUTER_INTERVALS_MOD_12 = frozenset({3, 4, 5, 7, 8, 9})


@dataclass(frozen=True)
class CounterpointGraphSpec:
    """Immutable contract for the graph G(n) traversed by the agent.

    This object defines the graph. It should stay small: node membership,
    edge predicates, action generation, and environment behavior should consume
    this spec rather than re-declaring its knobs.
    """

    n: int
    tonic: int
    pitch_min: int = 0
    pitch_max: int = 127
    max_interval: int = 11
    max_chord_width_factor: float = 5.0
    forbidden_adjacent_vertical_intervals: frozenset[int] = field(
        default_factory=lambda: DEFAULT_FORBIDDEN_ADJACENT_VERTICAL_INTERVALS
    )
    allowed_root_intervals_mod_12: frozenset[int] = field(
        default_factory=lambda: DEFAULT_ALLOWED_ROOT_INTERVALS_MOD_12
    )
    allowed_outer_intervals_mod_12: frozenset[int] = field(
        default_factory=lambda: DEFAULT_ALLOWED_OUTER_INTERVALS_MOD_12
    )
    allow_voice_crossing: bool = False
    allow_parallel_fifths: bool = False

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")

        if self.pitch_min > self.pitch_max:
            raise ValueError("pitch_min must be <= pitch_max")

        if self.pitch_count < self.n:
            raise ValueError("pitch range must contain at least n pitches")

        if self.max_interval < 1:
            raise ValueError("max_interval must be at least 1")

        if self.max_chord_width_factor < 0:
            raise ValueError("max_chord_width_factor must be non-negative")

        self._validate_interval_set(
            name="forbidden_adjacent_vertical_intervals",
            intervals=self.forbidden_adjacent_vertical_intervals,
            lower=0,
            upper=self.pitch_count - 1,
        )
        self._validate_interval_set(
            name="allowed_root_intervals_mod_12",
            intervals=self.allowed_root_intervals_mod_12,
            lower=0,
            upper=11,
        )
        self._validate_interval_set(
            name="allowed_outer_intervals_mod_12",
            intervals=self.allowed_outer_intervals_mod_12,
            lower=0,
            upper=11,
        )

    @property
    def pitch_count(self) -> int:
        return self.pitch_max - self.pitch_min + 1

    @property
    def tonic_pitch_class(self) -> int:
        return self.tonic % 12

    @property
    def max_chord_width(self) -> int:
        return ceil(self.n * self.max_chord_width_factor)

    @property
    def allowed_root_pitch_classes(self) -> frozenset[int]:
        return frozenset(
            (self.tonic_pitch_class + interval) % 12
            for interval in self.allowed_root_intervals_mod_12
        )

    @property
    def allowed_outer_interval_classes(self) -> frozenset[int]:
        return self.allowed_outer_intervals_mod_12

    @property
    def max_adjacent_vertical_interval(self) -> int:
        return self.max_interval

    @property
    def max_single_line_interval(self) -> int:
        return self.max_interval

    @staticmethod
    def _validate_interval_set(
        *,
        name: str,
        intervals: frozenset[int],
        lower: int,
        upper: int,
    ) -> None:
        if not intervals:
            raise ValueError(f"{name} must not be empty")

        invalid = [interval for interval in intervals if interval < lower or interval > upper]
        if invalid:
            raise ValueError(
                f"{name} contains values outside [{lower}, {upper}]: {invalid}"
            )
