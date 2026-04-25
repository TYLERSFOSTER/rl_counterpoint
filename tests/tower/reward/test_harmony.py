"""Tests for rank-2 harmonic reward terms."""

from __future__ import annotations

import pytest

from tower.reward.context import NewFacts, TowerRewardContext
from tower.reward.harmony import (
    Rank2CadenceEndpointReward,
    Rank2SpacingControlReward,
    Rank2TargetVerticalIntervalReward,
    Rank2VerticalConsonanceReward,
)
from tower.reward.melody import consonance_from_pitch_class
from tower.window import build_window


def make_context(
    *,
    history: tuple[tuple[int, ...], ...],
    action: tuple[int, ...],
    target_root_octave: int | None = None,
    new_voice_index: int | None = 1,
    key_pitch_class: int | None = 0,
    is_final_step: bool = False,
) -> TowerRewardContext:
    source = history[-1]
    target = tuple(pitch + delta for pitch, delta in zip(source, action, strict=True))
    step_index = len(history) - 1
    return TowerRewardContext(
        rank=2,
        step_index=step_index,
        source=source,
        target=target,
        action=action,
        window=build_window(
            history=history,
            step_index=step_index,
            measure_size=4,
            context_measures=2,
        ),
        measure_size=4,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
        is_final_step=is_final_step,
        new_facts=NewFacts(new_voice_index=new_voice_index),
    )


def test_rank2_vertical_consonance_reward_rewards_consonant_interval() -> None:
    term = Rank2VerticalConsonanceReward(consonance_weight=2.0)
    context = make_context(
        history=((60, 63),),
        action=(0, 1),
    )

    result = term(context)

    diagnostics = result.diagnostics["rank2_vertical_consonance"]
    interval = diagnostics["intervals"][0]
    assert result.reward == pytest.approx(2.0 * consonance_from_pitch_class(4))
    assert interval["interval_pitch_class"] == 4
    assert interval["is_consonant"] is True


def test_rank2_vertical_consonance_reward_penalizes_non_consonance() -> None:
    term = Rank2VerticalConsonanceReward(non_consonance_penalty=-3.0)
    context = make_context(
        history=((60, 62),),
        action=(0, 4),
    )

    result = term(context)

    diagnostics = result.diagnostics["rank2_vertical_consonance"]
    interval = diagnostics["intervals"][0]
    assert result.reward == -3.0
    assert interval["interval_pitch_class"] == 6
    assert interval["is_consonant"] is False


def test_rank2_harmonic_terms_reject_non_rank_2_context() -> None:
    context = TowerRewardContext(
        rank=1,
        step_index=0,
        source=(60,),
        target=(61,),
        action=(1,),
        window=build_window(
            history=((60,),),
            step_index=0,
            measure_size=4,
            context_measures=1,
        ),
    )

    with pytest.raises(ValueError, match="rank-2 harmonic reward requires rank 2"):
        Rank2VerticalConsonanceReward()(context)
    with pytest.raises(ValueError, match="rank-2 harmonic reward requires rank 2"):
        Rank2SpacingControlReward()(context)
    with pytest.raises(ValueError, match="rank-2 harmonic reward requires rank 2"):
        Rank2TargetVerticalIntervalReward()(context)


def test_rank2_spacing_control_rewards_safe_spacing_below_ceiling() -> None:
    term = Rank2SpacingControlReward(
        upper_register_soft_ceiling=76,
        min_vertical_gap=3,
        spacing_reward=0.25,
        spacing_penalty=-0.5,
    )

    result = term(
        make_context(
            history=((60, 64),),
            action=(0, 1),
        )
    )

    diagnostics = result.diagnostics["rank2_spacing_control"]
    assert result.reward == 0.25
    assert diagnostics["vertical_gap"] == 5
    assert diagnostics["excess_above_ceiling"] == 0


def test_rank2_spacing_control_penalizes_compressed_gap_and_ceiling_excess() -> None:
    term = Rank2SpacingControlReward(
        upper_register_soft_ceiling=76,
        upper_register_penalty_weight=0.25,
        min_vertical_gap=3,
        spacing_reward=0.25,
        spacing_penalty=-0.5,
    )

    result = term(
        make_context(
            history=((76, 77),),
            action=(0, 1),
        )
    )

    diagnostics = result.diagnostics["rank2_spacing_control"]
    assert result.reward == pytest.approx(-1.0)
    assert diagnostics["vertical_gap"] == 2
    assert diagnostics["excess_above_ceiling"] == 2
    assert diagnostics["ceiling_penalty"] == pytest.approx(-0.5)


def test_rank2_target_vertical_interval_reward_peaks_at_target_gap() -> None:
    term = Rank2TargetVerticalIntervalReward(
        target_vertical_interval=5,
        interval_reward_weight=1.0,
    )

    result = term(
        make_context(
            history=((60, 64),),
            action=(0, 1),
        )
    )

    diagnostics = result.diagnostics["rank2_target_vertical_interval"]
    assert result.reward == 1.0
    assert diagnostics["vertical_gap"] == 5
    assert diagnostics["interval_distance"] == 0


def test_rank2_target_vertical_interval_reward_decays_by_distance() -> None:
    term = Rank2TargetVerticalIntervalReward(
        target_vertical_interval=5,
        interval_reward_weight=2.0,
    )

    result = term(
        make_context(
            history=((60, 64),),
            action=(0, 3),
        )
    )

    diagnostics = result.diagnostics["rank2_target_vertical_interval"]
    assert diagnostics["vertical_gap"] == 7
    assert diagnostics["interval_distance"] == 2
    assert result.reward == pytest.approx(2.0 / 3.0)


def test_rank2_cadence_endpoint_reward_noops_before_final_step() -> None:
    term = Rank2CadenceEndpointReward(weight=2.0)

    result = term(
        make_context(
            history=((67, 71),),
            action=(-7, -7),
            key_pitch_class=0,
            is_final_step=False,
        )
    )

    assert result.reward == 0.0
    assert result.diagnostics["rank2_cadence_endpoint"]["reason"] == "not_final_step"


def test_rank2_cadence_endpoint_reward_scores_outer_third_closeness() -> None:
    term = Rank2CadenceEndpointReward(weight=2.0)

    result = term(
        make_context(
            history=((67, 71),),
            action=(-7, -7),
            key_pitch_class=0,
            is_final_step=True,
        )
    )

    diagnostics = result.diagnostics["rank2_cadence_endpoint"]
    assert result.reward == 2.0
    assert diagnostics["previous_distance"] == 0
    assert diagnostics["final_distance"] == 0
