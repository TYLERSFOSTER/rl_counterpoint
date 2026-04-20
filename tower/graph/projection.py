"""Tuple projections for tower states, actions, and windows."""

from __future__ import annotations

from tower.state_action import TowerAction, TowerState, validate_state
from tower.window import TowerWindow


def project_tuple(values: tuple[int, ...]) -> tuple[int, ...]:
    """Project one rank-k coordinate tuple to rank k - 1."""
    if not isinstance(values, tuple):
        raise TypeError("values must be a tuple")
    if len(values) < 2:
        raise ValueError("cannot project rank 1 tuple")
    if any(not isinstance(value, int) for value in values):
        raise TypeError("values entries must be ints")

    if len(values) == 2:
        return (values[0],)

    remove_index = len(values) - 2
    return values[:remove_index] + values[remove_index + 1 :]


def project_state(state: TowerState) -> TowerState:
    """Project one tower state to its parent state."""
    validate_state(state)
    return project_tuple(state)


def project_action(action: TowerAction) -> TowerAction:
    """Project one tower action to its parent action."""
    return project_tuple(action)


def project_window(window: TowerWindow) -> TowerWindow:
    """Project every state in a tower window while preserving meter and mask."""
    if not (
        len(window.states)
        == len(window.bar_positions)
        == len(window.valid_mask)
    ):
        raise ValueError("window fields must have the same length")
    if not window.states:
        raise ValueError("window must not be empty")

    return TowerWindow(
        states=tuple(project_tuple(state) for state in window.states),
        bar_positions=window.bar_positions,
        valid_mask=window.valid_mask,
    )
