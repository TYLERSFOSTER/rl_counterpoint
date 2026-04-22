"""Tests for rank-1 melodic reward shaping terms."""

from __future__ import annotations

import pytest

from tower.reward.context import TowerRewardContext
from tower.reward.melody import (
    BeatClassPitchReward,
    LargeLeapRecoveryTerm,
    RecentMelodicRangePenalty,
    StepSizeBinBalanceReward,
    TargetOctaveDistanceReward,
    consonance_from_pitch_class,
    midi_to_octave,
)
from tower.state_action import TowerAction, TowerState
from tower.window import build_window


def make_context(
    *,
    history: tuple[TowerState, ...],
    action: TowerAction,
    rank: int = 1,
    target_root_octave: int | None = None,
    key_pitch_class: int | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
) -> TowerRewardContext:
    source = history[-1]
    target = tuple(pitch + delta for pitch, delta in zip(source, action, strict=True))
    step_index = len(history) - 1
    return TowerRewardContext(
        rank=rank,
        step_index=step_index,
        source=source,
        target=target,
        action=action,
        window=build_window(
            history=history,
            step_index=step_index,
            measure_size=measure_size,
            context_measures=context_measures,
        ),
        measure_size=measure_size,
        key_pitch_class=key_pitch_class,
        target_root_octave=target_root_octave,
    )


def test_midi_to_octave_uses_scientific_pitch_octaves() -> None:
    assert midi_to_octave(0) == -1
    assert midi_to_octave(12) == 0
    assert midi_to_octave(60) == 4
    assert midi_to_octave(72) == 5
    assert midi_to_octave(127) == 9


def test_target_octave_distance_reward_uses_inverse_distance() -> None:
    term = TargetOctaveDistanceReward()
    context = make_context(
        history=((60,),),
        action=(12,),
        target_root_octave=4,
    )

    result = term(context)

    diagnostics = result.diagnostics["target_octave_distance"]
    assert result.reward == 0.5
    assert diagnostics["target_pitch"] == 72
    assert diagnostics["root_octave"] == 5
    assert diagnostics["target_root_octave"] == 4
    assert diagnostics["octave_distance"] == 1
    assert diagnostics["reward_formula"] == "1/(1+d)"


def test_target_octave_distance_reward_is_one_at_goal_octave() -> None:
    term = TargetOctaveDistanceReward()
    context = make_context(
        history=((60,),),
        action=(1,),
        target_root_octave=4,
    )

    result = term(context)

    assert result.reward == 1.0
    assert result.diagnostics["target_octave_distance"]["octave_distance"] == 0


def test_target_octave_distance_reward_requires_target_octave() -> None:
    term = TargetOctaveDistanceReward()
    context = make_context(history=((60,),), action=(1,))

    with pytest.raises(ValueError, match="target_root_octave is required"):
        term(context)


def test_beat_class_pitch_reward_rewards_tonic_on_measure_start() -> None:
    term = BeatClassPitchReward(
        measure_start_tonic_reward=1.25,
        onbeat_scale_degree_reward=0.75,
    )
    context = make_context(
        history=((59,),),
        action=(1,),
        key_pitch_class=0,
    )

    result = term(context)

    diagnostics = result.diagnostics["beat_class_pitch"]
    assert result.reward == 2.0
    assert diagnostics["bar_position"] == 0
    assert diagnostics["is_measure_start"] is True
    assert diagnostics["is_onbeat"] is True
    assert diagnostics["is_tonic"] is True
    assert diagnostics["is_scale_degree"] is True
    assert diagnostics["is_consonant"] is True
    assert diagnostics["measure_start_tonic_reward"] == 1.25
    assert diagnostics["onbeat_scale_degree_reward"] == 0.75
    assert diagnostics["onbeat_non_scale_penalty"] == 0.0
    assert diagnostics["offbeat_consonance_reward"] == 0.0
    assert diagnostics["offbeat_non_consonance_penalty"] == 0.0
    assert diagnostics["beat_class_timing"] == "source_window_step_index"


def test_beat_class_pitch_reward_rewards_major_scale_degree_on_onbeat() -> None:
    term = BeatClassPitchReward(onbeat_scale_degree_reward=0.75)
    context = make_context(
        history=((60,), (61,), (61,)),
        action=(1,),
        key_pitch_class=0,
    )

    result = term(context)

    diagnostics = result.diagnostics["beat_class_pitch"]
    assert result.reward == 0.75
    assert diagnostics["bar_position"] == 2
    assert diagnostics["is_measure_start"] is False
    assert diagnostics["is_onbeat"] is True
    assert diagnostics["relative_pitch_class"] == 2
    assert diagnostics["is_scale_degree"] is True


def test_beat_class_pitch_reward_penalizes_non_scale_onbeat() -> None:
    term = BeatClassPitchReward()
    context = make_context(
        history=((60,), (61,), (60,)),
        action=(1,),
        key_pitch_class=0,
    )

    result = term(context)

    diagnostics = result.diagnostics["beat_class_pitch"]
    assert result.reward == -2.0
    assert diagnostics["bar_position"] == 2
    assert diagnostics["relative_pitch_class"] == 1
    assert diagnostics["is_scale_degree"] is False
    assert diagnostics["onbeat_non_scale_penalty"] == -2.0


def test_beat_class_pitch_reward_rewards_just_consonance_on_offbeat() -> None:
    term = BeatClassPitchReward(offbeat_consonance_weight=2.0)
    context = make_context(
        history=((66,), (66,)),
        action=(1,),
        key_pitch_class=0,
    )

    result = term(context)

    diagnostics = result.diagnostics["beat_class_pitch"]
    assert result.reward == pytest.approx(2.0 * consonance_from_pitch_class(7))
    assert diagnostics["bar_position"] == 1
    assert diagnostics["is_offbeat"] is True
    assert diagnostics["relative_pitch_class"] == 7
    assert diagnostics["is_consonant"] is True
    assert diagnostics["just_consonance"] == pytest.approx(consonance_from_pitch_class(7))
    assert diagnostics["offbeat_consonance_reward"] == pytest.approx(
        2.0 * consonance_from_pitch_class(7)
    )
    assert diagnostics["offbeat_non_consonance_penalty"] == 0.0


def test_beat_class_pitch_reward_penalizes_non_consonant_offbeat() -> None:
    term = BeatClassPitchReward(offbeat_non_consonance_penalty=-3.0)
    context = make_context(
        history=((60,), (60,)),
        action=(1,),
        key_pitch_class=0,
    )

    result = term(context)

    diagnostics = result.diagnostics["beat_class_pitch"]
    assert result.reward == pytest.approx(consonance_from_pitch_class(1) - 3.0)
    assert diagnostics["bar_position"] == 1
    assert diagnostics["is_offbeat"] is True
    assert diagnostics["relative_pitch_class"] == 1
    assert diagnostics["is_consonant"] is False
    assert diagnostics["offbeat_non_consonance_penalty"] == -3.0


def test_beat_class_pitch_reward_requires_key_and_measure_size() -> None:
    term = BeatClassPitchReward()
    context = make_context(history=((60,),), action=(1,), key_pitch_class=None)

    with pytest.raises(ValueError, match="key_pitch_class is required"):
        term(context)

    context = make_context(history=((60,),), action=(1,), key_pitch_class=0)
    context = TowerRewardContext(
        rank=context.rank,
        step_index=context.step_index,
        source=context.source,
        target=context.target,
        action=context.action,
        window=context.window,
        measure_size=None,
        key_pitch_class=context.key_pitch_class,
    )

    with pytest.raises(ValueError, match="measure_size is required"):
        term(context)


def test_recent_melodic_range_penalty_fires_above_threshold() -> None:
    term = RecentMelodicRangePenalty(max_recent_range=12, penalty=-1.25)
    context = make_context(history=((60,), (73,)), action=(0,))

    result = term(context)

    diagnostics = result.diagnostics["recent_melodic_range"]
    assert result.reward == -1.25
    assert diagnostics["observed_range"] == 13
    assert diagnostics["max_recent_range"] == 12
    assert diagnostics["valid_pitch_count"] == 2
    assert diagnostics["penalty_applied"] is True
    assert diagnostics["reason"] == "range_exceeded"


def test_recent_melodic_range_penalty_allows_exact_threshold() -> None:
    term = RecentMelodicRangePenalty(max_recent_range=12)
    context = make_context(history=((60,), (72,)), action=(0,))

    result = term(context)

    diagnostics = result.diagnostics["recent_melodic_range"]
    assert result.reward == 0.0
    assert diagnostics["observed_range"] == 12
    assert diagnostics["penalty_applied"] is False
    assert diagnostics["reason"] == "within_range"


def test_recent_melodic_range_penalty_noops_with_short_history() -> None:
    term = RecentMelodicRangePenalty()
    context = make_context(history=((60,),), action=(0,))

    result = term(context)

    diagnostics = result.diagnostics["recent_melodic_range"]
    assert result.reward == 0.0
    assert diagnostics["observed_range"] is None
    assert diagnostics["valid_pitch_count"] == 1
    assert diagnostics["penalty_applied"] is False
    assert diagnostics["reason"] == "insufficient_valid_history"


def test_recent_melodic_range_penalty_rejects_non_rank_1_context() -> None:
    term = RecentMelodicRangePenalty()
    context = make_context(history=((60, 64), (62, 65)), action=(1, 1), rank=2)

    with pytest.raises(ValueError, match="require rank 1 context"):
        term(context)


def test_recent_melodic_range_penalty_validates_configuration() -> None:
    with pytest.raises(ValueError, match="max_recent_range must be non-negative"):
        RecentMelodicRangePenalty(max_recent_range=-1)


def test_large_leap_recovery_rewards_opposite_stepwise_motion() -> None:
    term = LargeLeapRecoveryTerm(
        large_leap_threshold=6,
        recovery_step_threshold=3,
        recovery_reward=0.75,
    )
    context = make_context(history=((60,), (67,)), action=(-2,))

    result = term(context)

    diagnostics = result.diagnostics["large_leap_recovery"]
    assert result.reward == 0.75
    assert diagnostics["previous_interval"] == 7
    assert diagnostics["current_action"] == -2
    assert diagnostics["triggered"] is True
    assert diagnostics["opposite_direction"] is True
    assert diagnostics["small_step"] is True
    assert diagnostics["success"] is True
    assert diagnostics["reason"] == "recovered"


def test_large_leap_recovery_penalizes_wrong_direction() -> None:
    term = LargeLeapRecoveryTerm(failure_penalty=-0.75)
    context = make_context(history=((60,), (67,)), action=(2,))

    result = term(context)

    diagnostics = result.diagnostics["large_leap_recovery"]
    assert result.reward == -0.75
    assert diagnostics["triggered"] is True
    assert diagnostics["opposite_direction"] is False
    assert diagnostics["small_step"] is True
    assert diagnostics["success"] is False
    assert diagnostics["reason"] == "failed_recovery"


def test_large_leap_recovery_penalizes_oversized_recovery_motion() -> None:
    term = LargeLeapRecoveryTerm(recovery_step_threshold=3, failure_penalty=-0.75)
    context = make_context(history=((60,), (67,)), action=(-4,))

    result = term(context)

    diagnostics = result.diagnostics["large_leap_recovery"]
    assert result.reward == -0.75
    assert diagnostics["triggered"] is True
    assert diagnostics["opposite_direction"] is True
    assert diagnostics["small_step"] is False
    assert diagnostics["success"] is False
    assert diagnostics["reason"] == "failed_recovery"


def test_large_leap_recovery_noops_without_prior_large_leap() -> None:
    term = LargeLeapRecoveryTerm(large_leap_threshold=6)
    context = make_context(history=((60,), (65,)), action=(-2,))

    result = term(context)

    diagnostics = result.diagnostics["large_leap_recovery"]
    assert result.reward == 0.0
    assert diagnostics["previous_interval"] == 5
    assert diagnostics["triggered"] is False
    assert diagnostics["reason"] == "no_large_leap"


def test_large_leap_recovery_noops_with_short_history() -> None:
    term = LargeLeapRecoveryTerm()
    context = make_context(history=((60,),), action=(0,))

    result = term(context)

    diagnostics = result.diagnostics["large_leap_recovery"]
    assert result.reward == 0.0
    assert diagnostics["previous_interval"] is None
    assert diagnostics["triggered"] is False
    assert diagnostics["reason"] == "insufficient_valid_history"


def test_large_leap_recovery_rejects_non_rank_1_context() -> None:
    term = LargeLeapRecoveryTerm()
    context = make_context(history=((60, 64), (67, 72)), action=(-2, 1), rank=2)

    with pytest.raises(ValueError, match="require rank 1 context"):
        term(context)


def test_large_leap_recovery_validates_configuration() -> None:
    with pytest.raises(ValueError, match="large_leap_threshold must be at least 1"):
        LargeLeapRecoveryTerm(large_leap_threshold=0)

    with pytest.raises(ValueError, match="recovery_step_threshold must be at least 1"):
        LargeLeapRecoveryTerm(recovery_step_threshold=0)


def test_step_size_bin_balance_rewards_default_thirty_seventy_target() -> None:
    term = StepSizeBinBalanceReward(small_step_threshold=3, weight=2.0)
    context = make_context(
        history=((60,), (61,), (62,), (66,), (70,), (74,), (78,), (82,), (86,), (90,)),
        action=(1,),
        context_measures=3,
    )

    result = term(context)

    diagnostics = result.diagnostics["step_size_bin_balance"]
    assert result.reward == 2.0
    assert diagnostics["small_count"] == 3
    assert diagnostics["large_count"] == 7
    assert diagnostics["target_small_rate"] == 0.3
    assert diagnostics["target_large_rate"] == 0.7
    assert diagnostics["observed_small_rate"] == 0.3
    assert diagnostics["observed_large_rate"] == 0.7
    assert diagnostics["balance_score"] == 1.0
    assert diagnostics["reason"] == "target_matched"


def test_step_size_bin_balance_can_target_equal_bins() -> None:
    term = StepSizeBinBalanceReward(
        small_step_threshold=3,
        target_small_rate=0.5,
        weight=2.0,
    )
    context = make_context(history=((60,), (62,)), action=(5,))

    result = term(context)

    diagnostics = result.diagnostics["step_size_bin_balance"]
    assert result.reward == 2.0
    assert diagnostics["small_count"] == 1
    assert diagnostics["large_count"] == 1
    assert diagnostics["balance_score"] == 1.0
    assert diagnostics["reason"] == "target_matched"


def test_step_size_bin_balance_partially_rewards_off_target_mix() -> None:
    term = StepSizeBinBalanceReward(small_step_threshold=3, weight=2.0)
    context = make_context(history=((60,), (62,)), action=(5,))

    result = term(context)

    diagnostics = result.diagnostics["step_size_bin_balance"]
    assert result.reward == pytest.approx(2.0 * (0.5 / 0.7))
    assert diagnostics["observed_small_rate"] == 0.5
    assert diagnostics["observed_large_rate"] == 0.5
    assert diagnostics["balance_score"] == pytest.approx(0.5 / 0.7)
    assert diagnostics["reason"] == "off_target"


def test_step_size_bin_balance_reward_is_zero_for_one_bin_only() -> None:
    term = StepSizeBinBalanceReward(small_step_threshold=3, weight=2.0)
    context = make_context(history=((60,), (62,), (64,)), action=(3,))

    result = term(context)

    diagnostics = result.diagnostics["step_size_bin_balance"]
    assert result.reward == 0.0
    assert diagnostics["small_count"] == 3
    assert diagnostics["large_count"] == 0
    assert diagnostics["balance_score"] == 0.0
    assert diagnostics["reason"] == "off_target"


def test_step_size_bin_balance_reward_noops_with_too_few_intervals() -> None:
    term = StepSizeBinBalanceReward()
    context = make_context(history=((60,),), action=(7,))

    result = term(context)

    diagnostics = result.diagnostics["step_size_bin_balance"]
    assert result.reward == 0.0
    assert diagnostics["small_count"] == 0
    assert diagnostics["large_count"] == 1
    assert diagnostics["observed_small_rate"] is None
    assert diagnostics["observed_large_rate"] is None
    assert diagnostics["balance_score"] == 0.0
    assert diagnostics["reason"] == "insufficient_intervals"


def test_step_size_bin_balance_rejects_non_rank_1_context() -> None:
    term = StepSizeBinBalanceReward()
    context = make_context(history=((60, 64), (62, 65)), action=(1, 1), rank=2)

    with pytest.raises(ValueError, match="require rank 1 context"):
        term(context)


def test_step_size_bin_balance_validates_configuration() -> None:
    with pytest.raises(ValueError, match="small_step_threshold must be at least 1"):
        StepSizeBinBalanceReward(small_step_threshold=0)

    with pytest.raises(TypeError, match="weight must be a real number"):
        StepSizeBinBalanceReward(weight=True)

    with pytest.raises(ValueError, match="target_small_rate must be between 0 and 1"):
        StepSizeBinBalanceReward(target_small_rate=1.0)
