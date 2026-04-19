"""Tuple-based state and action helpers for the tower model."""

from __future__ import annotations

from typing import TypeAlias

MidiPitch: TypeAlias = int
TowerState: TypeAlias = tuple[int, ...]
TowerAction: TypeAlias = tuple[int, ...]


def rank_of_state(state: TowerState) -> int:
    """Return the rank determined by one tower state."""
    validate_state(state)
    return len(state)


def validate_rank(rank: int) -> None:
    """Validate a tower rank value."""
    if not isinstance(rank, int):
        raise TypeError("rank must be an int")
    if rank < 1:
        raise ValueError("rank must be at least 1")


def validate_state(state: TowerState, *, rank: int | None = None) -> None:
    """Validate the canonical tuple-based tower state contract."""
    if not isinstance(state, tuple):
        raise TypeError("state must be a tuple")
    if not state:
        raise ValueError("state must not be empty")
    if rank is not None:
        validate_rank(rank)
        if len(state) != rank:
            raise ValueError("state length must match rank")
    for pitch in state:
        if not isinstance(pitch, int):
            raise TypeError("state entries must be ints")
        if pitch < 0 or pitch > 127:
            raise ValueError("state entries must be in [0, 127]")
    if any(state[index] >= state[index + 1] for index in range(len(state) - 1)):
        raise ValueError("state entries must be strictly increasing")


def validate_action(action: TowerAction, *, rank: int) -> None:
    """Validate the canonical tuple-based tower action contract."""
    validate_rank(rank)
    if not isinstance(action, tuple):
        raise TypeError("action must be a tuple")
    if len(action) != rank:
        raise ValueError("action length must match rank")
    for delta in action:
        if not isinstance(delta, int):
            raise TypeError("action entries must be ints")


def apply_action(state: TowerState, action: TowerAction) -> TowerState:
    """Apply an action vector to a state coordinatewise."""
    rank = rank_of_state(state)
    validate_action(action, rank=rank)
    return tuple(
        pitch + delta
        for pitch, delta in zip(state, action, strict=True)
    )
