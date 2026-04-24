"""Minimal tower graph specification shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from tower.state_action import validate_rank


@dataclass(frozen=True)
class TowerGraphSpec:
    """Minimal graph knobs needed by early tower slices."""

    rank: int
    pitch_min: int = 0
    pitch_max: int = 127
    max_step_size: int = 4
    induced_node_image: FrozenSet[tuple[int, ...]] | None = None
    induced_edge_image: FrozenSet[tuple[tuple[int, ...], tuple[int, ...]]] | None = None

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
        if self.induced_node_image is not None:
            if self.rank != 1:
                raise ValueError("induced_node_image currently requires rank 1")
            for state in self.induced_node_image:
                if not isinstance(state, tuple) or len(state) != 1:
                    raise ValueError("induced_node_image entries must be rank-1 tuples")
        if self.induced_edge_image is not None:
            if self.rank != 1:
                raise ValueError("induced_edge_image currently requires rank 1")
            for source, target in self.induced_edge_image:
                if (
                    not isinstance(source, tuple)
                    or not isinstance(target, tuple)
                    or len(source) != 1
                    or len(target) != 1
                ):
                    raise ValueError(
                        "induced_edge_image entries must be pairs of rank-1 tuples"
                    )
