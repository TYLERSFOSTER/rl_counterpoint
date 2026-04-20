"""Reward context contracts for tower rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from tower.state_action import TowerAction, TowerState, validate_action, validate_rank, validate_state
from tower.window import TowerWindow


@dataclass(frozen=True)
class NewFacts:
    """Rank-local facts introduced by the active tower rank."""

    new_voice_index: int | None = None
    new_action: int | None = None
    new_vertical_facts: tuple[int, ...] = ()
    full_sonority_used: bool = False


@dataclass(frozen=True)
class TowerRewardContext:
    """Structured rank-local input supplied to tower reward functions."""

    rank: int
    step_index: int
    source: TowerState
    target: TowerState
    action: TowerAction
    window: TowerWindow
    measure_size: int | None = None
    max_steps: int | None = None
    max_step_size: int | None = None
    key_pitch_class: int | None = None
    target_root_octave: int | None = None
    is_final_step: bool = False
    new_facts: NewFacts = field(default_factory=NewFacts)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")

        validate_state(self.source, rank=self.rank)
        validate_state(self.target, rank=self.rank)
        validate_action(self.action, rank=self.rank)
        _validate_window_for_rank(self.window, rank=self.rank)


def _validate_window_for_rank(window: TowerWindow, *, rank: int) -> None:
    if not (
        len(window.states)
        == len(window.bar_positions)
        == len(window.valid_mask)
    ):
        raise ValueError("window fields must have the same length")
    if not window.states:
        raise ValueError("window must not be empty")

    for state, is_valid in zip(window.states, window.valid_mask, strict=True):
        if is_valid:
            validate_state(state, rank=rank)
        elif len(state) != rank:
            raise ValueError("window padding state length must match rank")
