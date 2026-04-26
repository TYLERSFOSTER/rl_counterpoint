"""Tests for minimal tower graph legality."""

from __future__ import annotations

from tower.graph.legality import is_valid_state, is_valid_transition
from tower.graph.spec import TowerGraphSpec


def test_valid_rank_1_state() -> None:
    assert is_valid_state((60,), TowerGraphSpec(rank=1))


def test_valid_rank_2_increasing_state() -> None:
    assert is_valid_state((63, 67), TowerGraphSpec(rank=2))


def test_non_increasing_rank_2_state_false() -> None:
    assert not is_valid_state((60, 60), TowerGraphSpec(rank=2))


def test_rank_2_non_consonant_vertical_interval_false() -> None:
    assert not is_valid_state((60, 61), TowerGraphSpec(rank=2))


def test_rank_2_perfect_fourth_vertical_interval_false() -> None:
    assert not is_valid_state((63, 68), TowerGraphSpec(rank=2))


def test_rank_2_octave_vertical_interval_false() -> None:
    assert not is_valid_state((60, 72), TowerGraphSpec(rank=2))


def test_rank_2_lower_note_must_be_consonant_relative_to_key() -> None:
    assert is_valid_state((63, 67), TowerGraphSpec(rank=2, key_pitch_class=0))
    assert not is_valid_state((60, 64), TowerGraphSpec(rank=2, key_pitch_class=0))


def test_transition_to_out_of_range_target_false() -> None:
    spec = TowerGraphSpec(rank=1, pitch_min=0, pitch_max=127)

    assert not is_valid_transition((127,), (1,), spec)


def test_self_loop_action_false() -> None:
    assert not is_valid_transition((60,), (0,), TowerGraphSpec(rank=1))


def test_valid_action_to_valid_target_true() -> None:
    assert is_valid_transition((60,), (2,), TowerGraphSpec(rank=1))


def test_rank_1_induced_node_and_edge_images_constrain_legality() -> None:
    spec = TowerGraphSpec(
        rank=1,
        pitch_min=60,
        pitch_max=61,
        max_step_size=1,
        induced_node_image=frozenset({(60,), (61,)}),
        induced_edge_image=frozenset({((60,), (61,))}),
    )

    assert is_valid_state((60,), spec)
    assert is_valid_state((61,), spec)
    assert not is_valid_transition((61,), (-1,), spec)
    assert is_valid_transition((60,), (1,), spec)


def test_rank_2_voice_crossing_transition_false() -> None:
    assert not is_valid_transition((60, 64), (5, -1), TowerGraphSpec(rank=2))


def test_rank_2_parallel_fifth_transition_false() -> None:
    assert not is_valid_transition((60, 67), (1, 1), TowerGraphSpec(rank=2))


def test_rank_2_stationary_voice_transition_false() -> None:
    assert not is_valid_transition((63, 67), (1, 0), TowerGraphSpec(rank=2))
    assert not is_valid_transition((63, 67), (0, 1), TowerGraphSpec(rank=2))
