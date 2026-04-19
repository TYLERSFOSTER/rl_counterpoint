"""Tests for tuple-based tower state and action helpers."""

from __future__ import annotations

import pytest

from tower.state_action import (
    apply_action,
    rank_of_state,
    validate_action,
    validate_state,
)


def test_rank_of_rank_1_state() -> None:
    assert rank_of_state((60,)) == 1


def test_rank_of_rank_2_state() -> None:
    assert rank_of_state((60, 64)) == 2


def test_empty_state_rejected() -> None:
    with pytest.raises(ValueError, match="state must not be empty"):
        rank_of_state(())


def test_non_int_state_entry_rejected() -> None:
    with pytest.raises(TypeError, match="state entries must be ints"):
        validate_state((60, "64"))  # type: ignore[arg-type]


def test_midi_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"state entries must be in \[0, 127\]"):
        validate_state((128,))


def test_non_increasing_rank_2_state_rejected() -> None:
    with pytest.raises(ValueError, match="state entries must be strictly increasing"):
        validate_state((60, 60))


def test_valid_rank_1_action_accepted() -> None:
    validate_action((2,), rank=1)


def test_wrong_action_length_rejected() -> None:
    with pytest.raises(ValueError, match="action length must match rank"):
        validate_action((1, 2), rank=1)


def test_apply_rank_1_action() -> None:
    assert apply_action((60,), (2,)) == (62,)


def test_apply_rank_2_action() -> None:
    assert apply_action((60, 67), (1, -1)) == (61, 66)
