"""Upward action assembly helpers for the tower model."""

from __future__ import annotations

from tower.graph.projection import project_action
from tower.state_action import TowerAction, validate_action, validate_rank


def new_voice_index(*, rank: int) -> int:
    """Return the newly introduced coordinate index for a tower rank."""
    validate_rank(rank)
    if rank == 1:
        return 0
    if rank == 2:
        return 1
    return rank - 2


def assemble_action(
    *,
    rank: int,
    parent_action: TowerAction | None,
    new_action: int,
) -> TowerAction:
    """Assemble a rank-k action from parent action plus the new coordinate."""
    validate_rank(rank)
    if not isinstance(new_action, int):
        raise TypeError("new_action must be an int")

    if rank == 1:
        if parent_action is not None:
            raise ValueError("rank 1 action must not have a parent_action")
        return (new_action,)

    if parent_action is None:
        raise ValueError("rank greater than 1 requires parent_action")
    validate_action(parent_action, rank=rank - 1)

    insert_index = new_voice_index(rank=rank)
    action = parent_action[:insert_index] + (new_action,) + parent_action[insert_index:]
    validate_action_lift(action=action, parent_action=parent_action)
    return action


def validate_action_lift(
    *,
    action: TowerAction,
    parent_action: TowerAction,
) -> None:
    """Validate that an action projects to the supplied parent action."""
    if project_action(action) != parent_action:
        raise ValueError("action must project to parent_action")
