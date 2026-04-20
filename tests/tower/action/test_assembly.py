"""Tests for tower action assembly helpers."""

from __future__ import annotations

import pytest

from tower.action.assembly import (
    assemble_action,
    new_voice_index,
    validate_action_lift,
)
from tower.graph.projection import project_action


def test_new_voice_index_by_rank() -> None:
    assert new_voice_index(rank=1) == 0
    assert new_voice_index(rank=2) == 1
    assert new_voice_index(rank=3) == 1
    assert new_voice_index(rank=4) == 2


def test_rank_1_assembly() -> None:
    assert assemble_action(rank=1, parent_action=None, new_action=2) == (2,)


def test_rank_2_assembly() -> None:
    assert assemble_action(rank=2, parent_action=(1,), new_action=-1) == (1, -1)


def test_rank_3_assembly() -> None:
    assert assemble_action(rank=3, parent_action=(1, 3), new_action=2) == (1, 2, 3)


def test_rank_4_assembly() -> None:
    assert assemble_action(rank=4, parent_action=(1, 2, 4), new_action=3) == (
        1,
        2,
        3,
        4,
    )


def test_assembled_action_projects_to_parent() -> None:
    parent = (1, 3)
    action = assemble_action(rank=3, parent_action=parent, new_action=2)

    assert project_action(action) == parent


def test_invalid_parent_length_rejected() -> None:
    with pytest.raises(ValueError, match="action length must match rank"):
        assemble_action(rank=3, parent_action=(1,), new_action=2)


def test_rank_1_parent_action_rejected() -> None:
    with pytest.raises(ValueError, match="rank 1 action must not have a parent_action"):
        assemble_action(rank=1, parent_action=(1,), new_action=2)


def test_rank_greater_than_1_missing_parent_rejected() -> None:
    with pytest.raises(ValueError, match="rank greater than 1 requires parent_action"):
        assemble_action(rank=2, parent_action=None, new_action=1)


def test_validate_action_lift_rejects_projection_mismatch() -> None:
    with pytest.raises(ValueError, match="action must project to parent_action"):
        validate_action_lift(action=(1, 2, 3), parent_action=(1, 2))
