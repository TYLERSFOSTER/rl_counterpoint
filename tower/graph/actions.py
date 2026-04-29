"""Candidate action generation for tower graph helpers."""

from __future__ import annotations

from functools import lru_cache
from itertools import product

from tower.action.assembly import new_voice_index
from tower.graph.legality import is_valid_state, is_valid_transition
from tower.graph.projection import project_action
from tower.graph.spec import TowerGraphSpec
from tower.state_action import TowerAction, validate_rank


@lru_cache(maxsize=None)
def action_space(*, rank: int, max_step_size: int) -> tuple[TowerAction, ...]:
    """Return the nonzero bounded action lattice for one tower rank."""
    validate_rank(rank)
    if max_step_size < 1:
        raise ValueError("max_step_size must be at least 1")

    deltas = range(-max_step_size, max_step_size + 1)
    zero_action = (0,) * rank
    return tuple(
        action
        for action in product(deltas, repeat=rank)
        if action != zero_action
    )


@lru_cache(maxsize=None)
def lift_fiber_actions(
    *,
    state: tuple[int, ...],
    parent_action: TowerAction,
    spec: TowerGraphSpec,
) -> tuple[TowerAction, ...]:
    """Return legal rank-k actions lying over one parent action."""
    if spec.rank <= 1:
        raise ValueError("lift_fiber_actions requires rank greater than 1")
    if not is_valid_state(state, spec):
        raise ValueError("state must be valid for spec")
    if len(parent_action) != spec.rank - 1:
        raise ValueError("parent_action rank must be spec.rank - 1")

    return tuple(
        action
        for action in action_space(rank=spec.rank, max_step_size=spec.max_step_size)
        if project_action(action) == parent_action
        and is_valid_transition(state, action, spec)
    )


@lru_cache(maxsize=None)
def active_lift_choices(
    *,
    state: tuple[int, ...],
    parent_action: TowerAction,
    spec: TowerGraphSpec,
) -> tuple[int, ...]:
    """Return unique active new-coordinate choices from a legal lift fiber."""
    active_index = new_voice_index(rank=spec.rank)
    choices = []
    seen = set()

    for action in lift_fiber_actions(
        state=state,
        parent_action=parent_action,
        spec=spec,
    ):
        choice = action[active_index]
        if choice not in seen:
            choices.append(choice)
            seen.add(choice)

    return tuple(choices)


def has_empty_lift_fiber(
    *,
    state: tuple[int, ...],
    parent_action: TowerAction,
    spec: TowerGraphSpec,
) -> bool:
    """Return True iff no legal rank-k lift exists over parent_action."""
    return not lift_fiber_actions(
        state=state,
        parent_action=parent_action,
        spec=spec,
    )


@lru_cache(maxsize=None)
def legal_actions_for_state(
    *,
    state: tuple[int, ...],
    spec: TowerGraphSpec,
) -> tuple[TowerAction, ...]:
    """Return all legal non-self actions from one state under one graph spec."""
    if not is_valid_state(state, spec):
        raise ValueError("state must be valid for spec")

    return tuple(
        action
        for action in action_space(rank=spec.rank, max_step_size=spec.max_step_size)
        if is_valid_transition(state, action, spec)
    )
