"""Tests for tower success predicates."""

from __future__ import annotations

import pytest

from tower.reward.context import TowerRewardContext
from tower.reward.success import (
    SuccessResult,
    rank1_projected_cadence_success,
    rank2_lifted_cadence_success,
)
from tower.window import build_window


def make_context(
    *,
    rank: int,
    history: tuple[tuple[int, ...], ...] | None = None,
    source: tuple[int, ...] | None = None,
    target: tuple[int, ...] | None = None,
    action: tuple[int, ...] | None = None,
    step_index: int = 0,
    measure_size: int | None = 4,
    key_pitch_class: int | None = 0,
    is_final_step: bool = True,
) -> TowerRewardContext:
    if rank == 1:
        if history is None:
            history = ((67,), (60,))
        source = (67,) if source is None else source
        target = (60,) if target is None else target
        action = (-7,) if action is None else action
    elif rank == 2:
        if history is None:
            history = ((60, 64),)
        source = (67, 71) if source is None else source
        target = (60, 64) if target is None else target
        action = (-7, -7) if action is None else action
    else:
        raise ValueError("test helper only supports rank 1 or 2")

    return TowerRewardContext(
        rank=rank,
        step_index=0,
        source=source,
        target=target,
        action=action,
        window=build_window(
            history=history,
            step_index=step_index,
            measure_size=measure_size if measure_size is not None else 4,
            context_measures=1,
        ),
        measure_size=measure_size,
        key_pitch_class=key_pitch_class,
        is_final_step=is_final_step,
    )


def test_rank1_projected_cadence_success_detects_terminal_v_i_root_motion() -> None:
    result = rank1_projected_cadence_success(
        make_context(
            rank=1,
            history=((65,), (66,)),
            source=(67,),
            target=(60,),
            action=(-7,),
            step_index=3,
            measure_size=4,
            key_pitch_class=0,
            is_final_step=True,
        )
    )

    assert isinstance(result, SuccessResult)
    assert result.success is True
    assert result.diagnostics["reason"] == "success"


def test_rank1_success_diagnostics_include_rank_kind_and_pitch_classes() -> None:
    result = rank1_projected_cadence_success(
        make_context(rank=1, step_index=3)
    )

    assert result.diagnostics["rank"] == 1
    assert result.diagnostics["kind"] == "rank1_projected_cadence_success"
    assert result.diagnostics["previous_pitch_class"] == 7
    assert result.diagnostics["final_pitch_class"] == 0
    assert result.diagnostics["dominant_pitch_class"] == 7
    assert result.diagnostics["tonic_pitch_class"] == 0


def test_rank1_success_rejects_wrong_root_motion() -> None:
    result = rank1_projected_cadence_success(
        make_context(rank=1, source=(65,), target=(60,), action=(-5,), step_index=3)
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "wrong_root_motion"


def test_rank1_success_uses_context_source_and_target_not_window_frontier() -> None:
    result = rank1_projected_cadence_success(
        make_context(
            rank=1,
            history=((65,), (66,)),
            source=(67,),
            target=(60,),
            action=(-7,),
            step_index=3,
        )
    )

    assert result.success is True
    assert result.diagnostics["previous_pitch_class"] == 7
    assert result.diagnostics["final_pitch_class"] == 0


def test_rank1_success_requires_final_step() -> None:
    result = rank1_projected_cadence_success(
        make_context(rank=1, step_index=3, is_final_step=False)
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "not_final_step"


def test_rank1_success_requires_ending_beat() -> None:
    result = rank1_projected_cadence_success(
        make_context(rank=1, step_index=2)
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "wrong_metrical_position"


def test_rank1_success_requires_key_pitch_class() -> None:
    result = rank1_projected_cadence_success(
        make_context(rank=1, step_index=3, key_pitch_class=None)
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "missing_key_pitch_class"


def test_rank1_success_requires_measure_size() -> None:
    result = rank1_projected_cadence_success(
        make_context(rank=1, step_index=3, measure_size=None)
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "missing_measure_size"


def test_rank1_success_rejects_non_rank_1_context() -> None:
    with pytest.raises(
        ValueError,
        match="rank1_projected_cadence_success requires rank 1 context",
    ):
        rank1_projected_cadence_success(make_context(rank=2))


def test_rank2_lifted_success_requires_parent_success_and_outer_thirds() -> None:
    result = rank2_lifted_cadence_success(
        make_context(
            rank=2,
            history=((65, 69), (66, 70)),
            source=(67, 71),
            target=(60, 64),
            action=(-7, -7),
            step_index=3,
            measure_size=4,
            key_pitch_class=0,
            is_final_step=True,
        )
    )

    assert result.success is True
    assert result.diagnostics["reason"] == "success"
    assert result.diagnostics["previous_outer_pitch_class"] == 11
    assert result.diagnostics["final_outer_pitch_class"] == 4
    assert result.diagnostics["dominant_third_pitch_class"] == 11
    assert result.diagnostics["tonic_third_pitch_class"] == 4
    assert result.diagnostics["parent"]["reason"] == "success"


def test_rank2_lifted_success_rejects_parent_failure() -> None:
    result = rank2_lifted_cadence_success(
        make_context(
            rank=2,
            history=((67, 71), (60, 64)),
            source=(65, 69),
            target=(60, 64),
            action=(-5, -5),
            step_index=3,
            measure_size=4,
            key_pitch_class=0,
            is_final_step=True,
        )
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "parent_success_failed"
    assert result.diagnostics["parent"]["reason"] == "wrong_root_motion"


def test_rank2_lifted_success_rejects_wrong_dominant_outer_third() -> None:
    result = rank2_lifted_cadence_success(
        make_context(
            rank=2,
            history=((67, 72), (60, 64)),
            source=(67, 72),
            target=(60, 64),
            action=(-7, -8),
            step_index=3,
            measure_size=4,
            key_pitch_class=0,
            is_final_step=True,
        )
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "wrong_dominant_outer_third"


def test_rank2_lifted_success_rejects_wrong_tonic_outer_third() -> None:
    result = rank2_lifted_cadence_success(
        make_context(
            rank=2,
            history=((67, 71), (60, 67)),
            source=(67, 71),
            target=(60, 67),
            action=(-7, -4),
            step_index=3,
            measure_size=4,
            key_pitch_class=0,
            is_final_step=True,
        )
    )

    assert result.success is False
    assert result.diagnostics["reason"] == "wrong_tonic_outer_third"


def test_rank2_lifted_success_rejects_non_rank_2_context() -> None:
    with pytest.raises(
        ValueError,
        match="rank2_lifted_cadence_success requires rank 2 context",
    ):
        rank2_lifted_cadence_success(make_context(rank=1))
