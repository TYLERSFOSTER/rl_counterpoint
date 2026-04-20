"""Trajectory record contracts for tower rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, TypeAlias

import torch

from tower.reward.result import TowerRewardResult
from tower.state_action import (
TowerAction,
    TowerState,
    validate_action,
    validate_rank,
    validate_state,
)
from tower.window import TowerWindow

LogProb: TypeAlias = float | torch.Tensor

TRAJECTORY_OUTCOME_VALID = "valid"
TRAJECTORY_OUTCOME_INVALID_EXTENSION = "invalid_extension"
TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER = "empty_lift_fiber"
TRAJECTORY_OUTCOME_PARENT_FAILURE = "parent_failure"

TrajectoryOutcome: TypeAlias = Literal[
    "valid",
    "invalid_extension",
    "empty_lift_fiber",
    "parent_failure",
]


@dataclass(frozen=True)
class TowerTrajectoryStep:
    """One post-transition rollout record at a single active rank."""

    rank: int
    step_index: int
    source_state: TowerState
    window: TowerWindow
    parent_state: TowerState | None
    parent_action: TowerAction | None
    active_choice: int | None
    assembled_action: TowerAction
    attempted_target_state: TowerState
    realized_next_state: TowerState
    active_logprob: LogProb | None
    reward: TowerRewardResult
    terminated: bool
    truncated: bool
    outcome: TrajectoryOutcome
    parent_logprob: LogProb | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        if not isinstance(self.step_index, int):
            raise TypeError("step_index must be an int")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")

        validate_state(self.source_state, rank=self.rank)
        validate_action(self.assembled_action, rank=self.rank)
        validate_state(self.attempted_target_state, rank=self.rank)
        validate_state(self.realized_next_state, rank=self.rank)
        _validate_window_for_rank(self.window, rank=self.rank)

        if self.parent_state is None:
            if self.rank > 1:
                raise ValueError("parent_state is required for rank greater than 1")
        else:
            validate_state(self.parent_state, rank=self.rank - 1)

        if self.parent_action is None:
            if self.rank > 1:
                raise ValueError("parent_action is required for rank greater than 1")
        else:
            validate_action(self.parent_action, rank=self.rank - 1)

        if self.active_choice is not None and not isinstance(self.active_choice, int):
            raise TypeError("active_choice must be an int or None")
        _validate_logprob(self.active_logprob, field_name="active_logprob")
        _validate_logprob(self.parent_logprob, field_name="parent_logprob")
        if not isinstance(self.reward, TowerRewardResult):
            raise TypeError("reward must be a TowerRewardResult")
        if self.outcome not in {
            TRAJECTORY_OUTCOME_VALID,
            TRAJECTORY_OUTCOME_INVALID_EXTENSION,
            TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
            TRAJECTORY_OUTCOME_PARENT_FAILURE,
        }:
            raise ValueError("outcome is not a recognized trajectory outcome")


@dataclass(frozen=True)
class TowerTrajectory:
    """A rank-local sequence of tower trajectory steps."""

    steps: tuple[TowerTrajectoryStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple")
        if not self.steps:
            return

        rank = self.steps[0].rank
        for step in self.steps:
            if step.rank != rank:
                raise ValueError("trajectory steps must have the same rank")

    @property
    def rank(self) -> int | None:
        """Return the trajectory rank, or None for an empty trajectory."""
        if not self.steps:
            return None
        return self.steps[0].rank

    @property
    def initial_state(self) -> TowerState | None:
        """Return the first source state, or None for an empty trajectory."""
        if not self.steps:
            return None
        return self.steps[0].source_state

    @property
    def final_state(self) -> TowerState | None:
        """Return the last realized next state, or None for an empty trajectory."""
        if not self.steps:
            return None
        return self.steps[-1].realized_next_state

    @property
    def total_reward(self) -> float:
        """Return the scalar reward sum across all trajectory steps."""
        return sum(step.reward.reward for step in self.steps)


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


def _validate_logprob(value: LogProb | None, *, field_name: str) -> None:
    if value is None or isinstance(value, float):
        return
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{field_name} tensor must be scalar")
        return
    raise TypeError(f"{field_name} must be a float, scalar tensor, or None")

