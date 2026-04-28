"""Minimal tower graph specification shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from tower.state_action import validate_rank


@dataclass(frozen=True)
class TowerGraphSpec:
    """Minimal graph knobs needed by early tower slices."""

    rank: int
    key_pitch_class: int = 0
    pitch_min: int = 0
    pitch_max: int = 127
    max_step_size: int = 4
    induced_node_image: FrozenSet[tuple[int, ...]] | None = None
    induced_edge_image: FrozenSet[tuple[tuple[int, ...], tuple[int, ...]]] | None = None

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        if self.key_pitch_class < 0 or self.key_pitch_class > 11:
            raise ValueError("key_pitch_class must be in [0, 11]")
        if self.pitch_min < 0:
            raise ValueError("pitch_min must be at least 0")
        if self.pitch_max > 127:
            raise ValueError("pitch_max must be at most 127")
        if self.pitch_min > self.pitch_max:
            raise ValueError("pitch_min must be <= pitch_max")
        if self.max_step_size < 1:
            raise ValueError("max_step_size must be at least 1")
        if self.induced_node_image is not None:
            if self.rank not in {1, 2}:
                raise ValueError("induced_node_image currently requires rank 1 or rank 2")
            for state in self.induced_node_image:
                if not isinstance(state, tuple) or len(state) != self.rank:
                    raise ValueError(
                        "induced_node_image entries must be tuples matching spec rank"
                    )
        if self.induced_edge_image is not None:
            if self.rank not in {1, 2}:
                raise ValueError("induced_edge_image currently requires rank 1 or rank 2")
            for source, target in self.induced_edge_image:
                if (
                    not isinstance(source, tuple)
                    or not isinstance(target, tuple)
                    or len(source) != self.rank
                    or len(target) != self.rank
                ):
                    raise ValueError(
                        "induced_edge_image entries must be pairs of tuples matching spec rank"
                    )
