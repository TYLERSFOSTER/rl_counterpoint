"""Tests for tower reward context shell."""

from __future__ import annotations

import pytest

from tower.reward.context import NewFacts, TowerRewardContext
from tower.window import TowerWindow, build_window


def make_rank_1_window() -> TowerWindow:
    return build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )


def test_context_accepts_rank_1_fields() -> None:
    context = TowerRewardContext(
        rank=1,
        step_index=0,
        source=(60,),
        target=(62,),
        action=(2,),
        window=make_rank_1_window(),
    )

    assert context.rank == 1
    assert context.source == (60,)
    assert context.target == (62,)
    assert context.action == (2,)


def test_context_rejects_mismatched_source_rank() -> None:
    with pytest.raises(ValueError, match="state length must match rank"):
        TowerRewardContext(
            rank=1,
            step_index=0,
            source=(60, 64),
            target=(62,),
            action=(2,),
            window=make_rank_1_window(),
        )


def test_context_rejects_mismatched_action_rank() -> None:
    with pytest.raises(ValueError, match="action length must match rank"):
        TowerRewardContext(
            rank=1,
            step_index=0,
            source=(60,),
            target=(62,),
            action=(2, 1),
            window=make_rank_1_window(),
        )


def test_context_rejects_mismatched_target_rank() -> None:
    with pytest.raises(ValueError, match="state length must match rank"):
        TowerRewardContext(
            rank=1,
            step_index=0,
            source=(60,),
            target=(62, 65),
            action=(2,),
            window=make_rank_1_window(),
        )


def test_new_facts_defaults() -> None:
    context = TowerRewardContext(
        rank=1,
        step_index=0,
        source=(60,),
        target=(62,),
        action=(2,),
        window=make_rank_1_window(),
    )

    assert context.new_facts == NewFacts()
    assert context.new_facts.new_voice_index is None
    assert context.new_facts.new_action is None
    assert context.new_facts.new_vertical_facts == ()
    assert context.new_facts.full_sonority_used is False


def test_rank_2_context_preserves_new_facts_for_active_outer_voice() -> None:
    new_facts = NewFacts(
        new_voice_index=1,
        new_action=1,
        new_vertical_facts=(4, 7),
        full_sonority_used=False,
    )

    context = TowerRewardContext(
        rank=2,
        step_index=3,
        source=(60, 64),
        target=(61, 65),
        action=(1, 1),
        window=build_window(
            history=((60, 64), (61, 65)),
            step_index=3,
            measure_size=4,
            context_measures=1,
        ),
        new_facts=new_facts,
        diagnostics={"new_facts_count": 2},
    )

    assert context.new_facts is new_facts
    assert context.new_facts.new_voice_index == 1
    assert context.new_facts.new_action == 1
    assert context.new_facts.new_vertical_facts == (4, 7)
    assert context.new_facts.full_sonority_used is False
    assert context.diagnostics["new_facts_count"] == 2


def test_metadata_fields_accepted() -> None:
    diagnostics = {"kind": "context_test"}
    context = TowerRewardContext(
        rank=1,
        step_index=3,
        source=(60,),
        target=(61,),
        action=(1,),
        window=make_rank_1_window(),
        measure_size=4,
        max_steps=16,
        max_step_size=2,
        key_pitch_class=0,
        target_root_octave=4,
        is_final_step=True,
        diagnostics=diagnostics,
    )

    assert context.measure_size == 4
    assert context.max_steps == 16
    assert context.max_step_size == 2
    assert context.key_pitch_class == 0
    assert context.target_root_octave == 4
    assert context.is_final_step is True
    assert context.diagnostics == diagnostics


def test_context_rejects_window_valid_state_rank_mismatch() -> None:
    window = TowerWindow(
        states=((0,), (60, 64)),
        bar_positions=(-1, 0),
        valid_mask=(False, True),
    )

    with pytest.raises(ValueError, match="state length must match rank"):
        TowerRewardContext(
            rank=1,
            step_index=0,
            source=(60,),
            target=(62,),
            action=(2,),
            window=window,
        )


def test_context_allows_masked_padding_state() -> None:
    window = TowerWindow(
        states=((0, 0), (60, 64)),
        bar_positions=(-1, 0),
        valid_mask=(False, True),
    )

    context = TowerRewardContext(
        rank=2,
        step_index=0,
        source=(60, 64),
        target=(61, 65),
        action=(1, 1),
        window=window,
    )

    assert context.window == window
