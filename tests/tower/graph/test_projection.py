"""Tests for tower projection helpers."""

from __future__ import annotations

import pytest

from tower.graph.projection import (
    project_action,
    project_state,
    project_tuple,
    project_window,
)
from tower.state_action import apply_action
from tower.window import TowerWindow


def test_project_rank_1_rejected() -> None:
    with pytest.raises(ValueError, match="cannot project rank 1 tuple"):
        project_tuple((60,))


def test_project_rank_2_state() -> None:
    assert project_state((60, 67)) == (60,)


def test_project_rank_3_state() -> None:
    assert project_state((60, 64, 67)) == (60, 67)


def test_project_rank_4_state() -> None:
    assert project_state((60, 64, 67, 72)) == (60, 64, 72)


def test_action_projection_mirrors_state_projection() -> None:
    assert project_action((1, -1)) == (1,)
    assert project_action((1, 2, 3)) == (1, 3)
    assert project_action((1, 2, 3, 4)) == (1, 2, 4)


def test_window_projection_preserves_bar_positions_and_valid_mask() -> None:
    window = TowerWindow(
        states=((0, 0), (60, 67), (62, 69)),
        bar_positions=(-1, 0, 1),
        valid_mask=(False, True, True),
        episode_step_indices=(-1, 4, 5),
    )

    projected = project_window(window)

    assert projected.states == ((0,), (60,), (62,))
    assert projected.bar_positions == window.bar_positions
    assert projected.valid_mask == window.valid_mask
    assert projected.episode_step_indices == window.episode_step_indices


def test_rank_2_projection_commutes_with_update() -> None:
    source = (60, 67)
    action = (2, -1)
    target = apply_action(source, action)

    assert project_state(target) == apply_action(
        project_state(source),
        project_action(action),
    )


def test_rank_3_projection_commutes_with_update() -> None:
    source = (60, 64, 67)
    action = (1, -1, 2)
    target = apply_action(source, action)

    assert project_state(target) == apply_action(
        project_state(source),
        project_action(action),
    )
