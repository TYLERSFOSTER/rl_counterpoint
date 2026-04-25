"""Tests for tower graph action helpers."""

from __future__ import annotations

import pytest

from tower.action.assembly import assemble_action
from tower.graph.actions import (
    action_space,
    active_lift_choices,
    has_empty_lift_fiber,
    lift_fiber_actions,
)
from tower.graph.legality import is_valid_transition
from tower.graph.projection import project_action
from tower.graph.spec import TowerGraphSpec


def test_rank_1_max_step_1_action_space() -> None:
    assert action_space(rank=1, max_step_size=1) == ((-1,), (1,))


def test_rank_2_max_step_1_action_space_has_8_actions() -> None:
    actions = action_space(rank=2, max_step_size=1)

    assert len(actions) == 8


def test_all_zero_action_excluded() -> None:
    actions = action_space(rank=3, max_step_size=1)

    assert (0, 0, 0) not in actions


def test_invalid_rank_rejected() -> None:
    with pytest.raises(ValueError, match="rank must be at least 1"):
        action_space(rank=0, max_step_size=1)


def test_invalid_max_step_rejected() -> None:
    with pytest.raises(ValueError, match="max_step_size must be at least 1"):
        action_space(rank=1, max_step_size=0)


def test_rank_2_fiber_over_parent_action() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)
    actions = lift_fiber_actions(
        state=(63, 67),
        parent_action=(1,),
        spec=spec,
    )

    assert actions
    assert all(project_action(action) == (1,) for action in actions)


def test_rank_2_fiber_excludes_other_parent_actions() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)
    actions = lift_fiber_actions(
        state=(63, 67),
        parent_action=(1,),
        spec=spec,
    )

    assert all(project_action(action) != (-1,) for action in actions)
    assert all(project_action(action) != (0,) for action in actions)


def test_lift_fiber_applies_legality_filter() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)
    actions = lift_fiber_actions(
        state=(63, 67),
        parent_action=(1,),
        spec=spec,
    )

    assert all(is_valid_transition((63, 67), action, spec) for action in actions)


def test_lift_fiber_excludes_stationary_rank_2_voices() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)
    actions = lift_fiber_actions(
        state=(63, 67),
        parent_action=(1,),
        spec=spec,
    )

    assert actions
    assert all(0 not in action for action in actions)


def test_lift_fiber_parent_rank_mismatch_rejected() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)

    with pytest.raises(ValueError, match="parent_action rank must be spec.rank - 1"):
        lift_fiber_actions(
            state=(63, 67),
            parent_action=(1, 2),
            spec=spec,
        )


def test_active_choices_align_with_fiber() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)
    parent_action = (1,)
    fiber = lift_fiber_actions(
        state=(63, 67),
        parent_action=parent_action,
        spec=spec,
    )
    choices = active_lift_choices(
        state=(63, 67),
        parent_action=parent_action,
        spec=spec,
    )

    assert choices
    for choice in choices:
        assert assemble_action(
            rank=spec.rank,
            parent_action=parent_action,
            new_action=choice,
        ) in fiber


def test_active_choices_are_unique() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)
    choices = active_lift_choices(
        state=(63, 67),
        parent_action=(1,),
        spec=spec,
    )

    assert len(choices) == len(set(choices))


def test_empty_fiber_gives_empty_active_choices() -> None:
    spec = TowerGraphSpec(
        rank=2,
        pitch_min=63,
        pitch_max=66,
        max_step_size=1,
    )

    assert active_lift_choices(
        state=(63, 66),
        parent_action=(1,),
        spec=spec,
    ) == ()


def test_nonempty_fiber_diagnostic_false() -> None:
    spec = TowerGraphSpec(rank=2, max_step_size=1)

    assert not has_empty_lift_fiber(
        state=(63, 67),
        parent_action=(1,),
        spec=spec,
    )


def test_empty_fiber_diagnostic_true() -> None:
    spec = TowerGraphSpec(
        rank=2,
        pitch_min=63,
        pitch_max=66,
        max_step_size=1,
    )

    assert has_empty_lift_fiber(
        state=(63, 66),
        parent_action=(1,),
        spec=spec,
    )
