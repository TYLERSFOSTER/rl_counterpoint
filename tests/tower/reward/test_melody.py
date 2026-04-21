"""Tests for rank-1 melodic reward shaping terms."""

from __future__ import annotations

import pytest

from tower.reward.context import TowerRewardContext
from tower.reward.melody import (
    LargeLeapRecoveryTerm,
    RecentMelodicRangePenalty,
    TargetOctaveDistanceReward,
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
            measure_size=4,
            context_measures=2,
        ),
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
