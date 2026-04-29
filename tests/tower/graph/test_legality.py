"""Tests for minimal tower graph legality."""

from __future__ import annotations

from tower.graph.legality import is_valid_state, is_valid_transition
from tower.graph.spec import TowerGraphSpec


def test_valid_rank_1_state() -> None:
    assert is_valid_state((60,), TowerGraphSpec(rank=1))


def test_rank_1_non_diatonic_state_false() -> None:
    assert not is_valid_state((61,), TowerGraphSpec(rank=1, key_pitch_class=0))


def test_valid_rank_2_increasing_state() -> None:
    assert is_valid_state((64, 68), TowerGraphSpec(rank=2))


def test_non_increasing_rank_2_state_false() -> None:
    assert not is_valid_state((60, 60), TowerGraphSpec(rank=2))


def test_rank_2_non_consonant_vertical_interval_false() -> None:
    assert not is_valid_state((60, 61), TowerGraphSpec(rank=2))


def test_rank_2_perfect_fourth_vertical_interval_false() -> None:
    assert not is_valid_state((63, 68), TowerGraphSpec(rank=2))


def test_rank_2_tritone_vertical_interval_true() -> None:
    assert is_valid_state((60, 66), TowerGraphSpec(rank=2))


def test_rank_2_minor_seventh_vertical_interval_true() -> None:
    assert is_valid_state((60, 70), TowerGraphSpec(rank=2))


def test_rank_2_octave_vertical_interval_false() -> None:
    assert not is_valid_state((60, 72), TowerGraphSpec(rank=2))


def test_rank_2_lower_note_must_project_to_valid_rank_1_state() -> None:
    assert is_valid_state((64, 68), TowerGraphSpec(rank=2, key_pitch_class=0))
    assert not is_valid_state((63, 67), TowerGraphSpec(rank=2, key_pitch_class=0))


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
        key_pitch_class=0,
        pitch_min=60,
        pitch_max=62,
        max_step_size=1,
        induced_node_image=frozenset({(60,), (61,), (62,)}),
        induced_edge_image=frozenset({((60,), (62,))}),
    )

    assert is_valid_state((60,), spec)
    assert not is_valid_state((61,), spec)
    assert is_valid_state((62,), spec)
    assert not is_valid_transition((61,), (-1,), spec)
    assert is_valid_transition((60,), (2,), spec)


def test_rank_2_voice_crossing_transition_false() -> None:
    assert not is_valid_transition((60, 64), (5, -1), TowerGraphSpec(rank=2))


def test_rank_2_parallel_fifth_transition_false() -> None:
    assert not is_valid_transition((60, 67), (1, 1), TowerGraphSpec(rank=2))


def test_rank_2_stationary_voice_transition_false() -> None:
    assert not is_valid_transition((64, 68), (1, 0), TowerGraphSpec(rank=2))
    assert not is_valid_transition((64, 68), (0, 1), TowerGraphSpec(rank=2))


def test_rank_2_induced_node_and_edge_images_constrain_legality() -> None:
    spec = TowerGraphSpec(
        rank=2,
        key_pitch_class=0,
        pitch_min=60,
        pitch_max=69,
        max_step_size=2,
        induced_node_image=frozenset({(60, 68), (62, 69)}),
        induced_edge_image=frozenset({((60, 68), (62, 69))}),
    )

    assert is_valid_state((60, 68), spec)
    assert not is_valid_state((60, 67), spec)
    assert is_valid_state((62, 69), spec)
    assert is_valid_transition((60, 68), (2, 1), spec)
    assert not is_valid_transition((62, 69), (-2, -1), spec)


def test_valid_rank_3_triad_state() -> None:
    assert is_valid_state((60, 64, 68), TowerGraphSpec(rank=3, key_pitch_class=0))


def test_rank_3_lower_voice_must_project_to_valid_rank_1_state() -> None:
    assert not is_valid_state((61, 64, 68), TowerGraphSpec(rank=3, key_pitch_class=0))


def test_rank_3_outer_pair_must_project_to_valid_rank_2_state() -> None:
    assert not is_valid_state((60, 64, 65), TowerGraphSpec(rank=3, key_pitch_class=0))


def test_rank_3_adjacent_interval_must_be_allowed() -> None:
    assert not is_valid_state((60, 65, 68), TowerGraphSpec(rank=3, key_pitch_class=0))


def test_rank_3_outer_interval_class_set_is_separate_and_allows_new_projected_rank_2_spans() -> None:
    assert is_valid_state((60, 63, 66), TowerGraphSpec(rank=3, key_pitch_class=0))


def test_rank_3_outer_width_must_be_bounded() -> None:
    assert not is_valid_state((60, 67, 76), TowerGraphSpec(rank=3, key_pitch_class=0))


def test_rank_3_valid_transition_true() -> None:
    spec = TowerGraphSpec(rank=3, key_pitch_class=0, pitch_min=60, pitch_max=69, max_step_size=2)

    assert is_valid_transition((60, 64, 68), (2, 1, 1), spec)


def test_rank_3_stationary_voice_transition_false() -> None:
    spec = TowerGraphSpec(rank=3, key_pitch_class=0, pitch_min=60, pitch_max=69, max_step_size=2)

    assert not is_valid_transition((60, 64, 68), (2, 0, 1), spec)


def test_rank_3_parallel_fifth_transition_false() -> None:
    spec = TowerGraphSpec(rank=3, key_pitch_class=0, pitch_min=60, pitch_max=69, max_step_size=1)

    assert not is_valid_transition((60, 64, 67), (1, 1, 1), spec)
