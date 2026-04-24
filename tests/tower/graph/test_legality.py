"""Tests for minimal tower graph legality."""

from __future__ import annotations

from tower.graph.legality import is_valid_state, is_valid_transition
from tower.graph.spec import TowerGraphSpec


def test_valid_rank_1_state() -> None:
    assert is_valid_state((60,), TowerGraphSpec(rank=1))


def test_valid_rank_2_increasing_state() -> None:
    assert is_valid_state((60, 64), TowerGraphSpec(rank=2))


def test_non_increasing_rank_2_state_false() -> None:
    assert not is_valid_state((60, 60), TowerGraphSpec(rank=2))


def test_rank_2_non_consonant_vertical_interval_false() -> None:
    assert not is_valid_state((60, 61), TowerGraphSpec(rank=2))


def test_rank_2_octave_vertical_interval_false() -> None:
    assert not is_valid_state((60, 72), TowerGraphSpec(rank=2))


def test_transition_to_out_of_range_target_false() -> None:
    spec = TowerGraphSpec(rank=1, pitch_min=0, pitch_max=127)

    assert not is_valid_transition((127,), (1,), spec)


def test_self_loop_action_false() -> None:
    assert not is_valid_transition((60,), (0,), TowerGraphSpec(rank=1))


def test_valid_action_to_valid_target_true() -> None:
    assert is_valid_transition((60,), (2,), TowerGraphSpec(rank=1))


def test_rank_2_voice_crossing_transition_false() -> None:
    assert not is_valid_transition((60, 64), (5, -1), TowerGraphSpec(rank=2))


def test_rank_2_parallel_fifth_transition_false() -> None:
    assert not is_valid_transition((60, 67), (1, 1), TowerGraphSpec(rank=2))
