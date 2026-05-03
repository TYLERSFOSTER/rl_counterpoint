"""Tests for tower history window helpers."""

from __future__ import annotations

import pytest

from tower.window import (
    PAD_BAR_POSITION,
    TowerWindow,
    bar_position,
    build_window,
    frontier_state,
    is_downbeat,
    is_ending_beat,
    pad_state,
)


def test_pad_rank_1() -> None:
    assert pad_state(rank=1) == (0,)


def test_pad_rank_2() -> None:
    assert pad_state(rank=2) == (0, 0)


def test_bar_position() -> None:
    assert bar_position(step_index=5, measure_size=4) == 1


def test_downbeat() -> None:
    assert is_downbeat(step_index=0)
    assert not is_downbeat(step_index=1)
    assert is_downbeat(step_index=2)


def test_ending_beat() -> None:
    assert is_ending_beat(step_index=3, measure_size=4)
    assert not is_ending_beat(step_index=2, measure_size=4)


def test_build_window_left_pads_short_history() -> None:
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=2,
    )

    assert isinstance(window, TowerWindow)
    assert len(window.states) == 8
    assert window.states[-1] == (60,)
    assert all(state == (0,) for state in window.states[:-1])
    assert window.bar_positions == (PAD_BAR_POSITION,) * 7 + (0,)
    assert window.valid_mask == (False,) * 7 + (True,)
    assert window.episode_step_indices == (-1,) * 7 + (0,)


def test_frontier_state_returns_final_valid_state() -> None:
    window = build_window(
        history=((60,), (62,), (64,)),
        step_index=2,
        measure_size=4,
        context_measures=2,
    )

    assert frontier_state(window) == (64,)


def test_frontier_state_ignores_final_padding_if_present() -> None:
    window = TowerWindow(
        states=((0,), (60,), (62,), (0,)),
        bar_positions=(PAD_BAR_POSITION, 0, 1, PAD_BAR_POSITION),
        valid_mask=(False, True, True, False),
    )

    assert frontier_state(window) == (62,)


def test_frontier_state_rejects_all_padding_window() -> None:
    window = TowerWindow(
        states=((0,), (0,)),
        bar_positions=(PAD_BAR_POSITION, PAD_BAR_POSITION),
        valid_mask=(False, False),
    )

    with pytest.raises(ValueError, match="at least one valid state"):
        frontier_state(window)


def test_build_window_records_real_bar_positions() -> None:
    window = build_window(
        history=((60,), (61,), (62,)),
        step_index=6,
        measure_size=4,
        context_measures=1,
    )

    assert window.states == ((0,), (60,), (61,), (62,))
    assert window.bar_positions == (PAD_BAR_POSITION, 0, 1, 2)
    assert window.valid_mask == (False, True, True, True)
    assert window.episode_step_indices == (-1, 4, 5, 6)


def test_build_window_truncates_long_history_to_recent_suffix() -> None:
    history = tuple((pitch,) for pitch in range(60, 70))

    window = build_window(
        history=history,
        step_index=9,
        measure_size=4,
        context_measures=2,
    )

    assert len(window.states) == 8
    assert window.states == history[-8:]
    assert window.bar_positions == (2, 3, 0, 1, 2, 3, 0, 1)
    assert window.valid_mask == (True,) * 8
    assert window.episode_step_indices == (2, 3, 4, 5, 6, 7, 8, 9)


def test_build_window_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="history must not be empty"):
        build_window(
            history=(),
            step_index=0,
            measure_size=4,
            context_measures=1,
        )


def test_build_window_rejects_nonpositive_measure_size() -> None:
    with pytest.raises(ValueError, match="measure_size must be at least 1"):
        build_window(
            history=((60,),),
            step_index=0,
            measure_size=0,
            context_measures=1,
        )


def test_build_window_rejects_nonpositive_context_measures() -> None:
    with pytest.raises(ValueError, match="context_measures must be at least 1"):
        build_window(
            history=((60,),),
            step_index=0,
            measure_size=4,
            context_measures=0,
        )


def test_build_window_rejects_mixed_rank_history() -> None:
    with pytest.raises(ValueError, match="state length must match rank"):
        build_window(
            history=((60,), (60, 64)),
            step_index=1,
            measure_size=4,
            context_measures=1,
        )
