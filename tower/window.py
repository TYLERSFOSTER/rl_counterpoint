"""Rank-local fixed-length history windows for the tower model."""

from __future__ import annotations

from dataclasses import dataclass

from tower.state_action import TowerState, rank_of_state, validate_rank, validate_state

PAD_BAR_POSITION = -1


@dataclass(frozen=True)
class TowerWindow:
    """Fixed-length padded rank-local state window."""

    states: tuple[TowerState, ...]
    bar_positions: tuple[int, ...]
    valid_mask: tuple[bool, ...]


def pad_state(*, rank: int) -> TowerState:
    """Return the distinguished PAD state for a tower rank."""
    validate_rank(rank)
    return (0,) * rank


def frontier_state(window: TowerWindow) -> TowerState:
    """Return the final valid state in a tower window."""
    if not isinstance(window, TowerWindow):
        raise TypeError("window must be a TowerWindow")
    if not window.states:
        raise ValueError("window.states must not be empty")
    if not (
        len(window.states) == len(window.valid_mask) == len(window.bar_positions)
    ):
        raise ValueError("window fields must have the same length")

    for state, is_valid in reversed(
        tuple(zip(window.states, window.valid_mask, strict=True))
    ):
        if is_valid:
            return state
    raise ValueError("window must contain at least one valid state")


def bar_position(*, step_index: int, measure_size: int) -> int:
    """Return the position of one step inside the current measure."""
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")
    return step_index % measure_size


def is_downbeat(*, step_index: int) -> bool:
    """Return True iff the current quarter-note step is even-indexed."""
    return step_index % 2 == 0


def is_ending_beat(*, step_index: int, measure_size: int) -> bool:
    """Return True iff the current step is the final beat in its measure."""
    return bar_position(step_index=step_index, measure_size=measure_size) == measure_size - 1


def build_window(
    *,
    history: tuple[TowerState, ...],
    step_index: int,
    measure_size: int,
    context_measures: int,
) -> TowerWindow:
    """Build a fixed-length left-padded tower window from raw history."""
    if not history:
        raise ValueError("history must not be empty")
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")
    if context_measures < 1:
        raise ValueError("context_measures must be at least 1")

    rank = rank_of_state(history[0])
    for state in history:
        validate_state(state, rank=rank)

    window_length = context_measures * measure_size
    real_history = history[-window_length:]
    real_start_step = step_index - len(real_history) + 1
    padding_length = window_length - len(real_history)
    padding_state = pad_state(rank=rank)

    states = (padding_state,) * padding_length + real_history
    bar_positions = (PAD_BAR_POSITION,) * padding_length + tuple(
        bar_position(step_index=real_start_step + offset, measure_size=measure_size)
        for offset in range(len(real_history))
    )
    valid_mask = (False,) * padding_length + (True,) * len(real_history)

    return TowerWindow(
        states=states,
        bar_positions=bar_positions,
        valid_mask=valid_mask,
    )
