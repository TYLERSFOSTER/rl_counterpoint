"""Minimal tower graph specification shell."""

from __future__ import annotations

from dataclasses import dataclass

from tower.state_action import validate_rank


@dataclass(frozen=True)
class TowerGraphSpec:
    """Minimal graph knobs needed by early tower slices."""

    rank: int
    pitch_min: int = 0
    pitch_max: int = 127
    max_step_size: int = 4

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        if self.pitch_min < 0:
            raise ValueError("pitch_min must be at least 0")
        if self.pitch_max > 127:
            raise ValueError("pitch_max must be at most 127")
        if self.pitch_min > self.pitch_max:
            raise ValueError("pitch_min must be <= pitch_max")
        if self.max_step_size < 1:
            raise ValueError("max_step_size must be at least 1")
