"""Tests for temporary black-box reward implementations.

These tests cover placeholder rewards used before the TC21M evaluator exists.
They verify that the placeholder obeys the reward protocol and exposes enough
diagnostic information for environment plumbing and smoke tests.
"""

from __future__ import annotations

from rl_counterpoint.envs.observation import TimedChordWindow
from rl_counterpoint.reward.black_box import BeatRoleDiagnosticReward, ConstantReward
from rl_counterpoint.reward.protocol import RewardContext, RewardResult


def test_constant_reward_returns_reward_result() -> None:
    """ConstantReward returns the structured RewardResult type."""
    reward_fn = ConstantReward(reward=1.25)

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=0))

    assert isinstance(result, RewardResult)


def test_constant_reward_returns_configured_reward() -> None:
    """ConstantReward returns the fixed scalar value it was configured with."""
    reward_fn = ConstantReward(reward=-0.5)

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=0))

    assert result.reward == -0.5
    assert not result.hard_violation
    assert not result.is_terminal_success


def test_constant_reward_includes_transition_diagnostics() -> None:
    """ConstantReward records source, target, step index, and placeholder kind."""
    reward_fn = ConstantReward(reward=0.0)

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=5))

    assert result.diagnostics["kind"] == "constant"
    assert result.diagnostics["source"] == (3, 6)
    assert result.diagnostics["target"] == (4, 7)
    assert result.diagnostics["step_index"] == 5


def test_constant_reward_merges_custom_diagnostics() -> None:
    """Custom diagnostics are included in the result diagnostics mapping."""
    reward_fn = ConstantReward(
        reward=0.0,
        diagnostics={"note": "plumbing-only", "kind": "custom"},
    )

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=0))

    assert result.diagnostics["kind"] == "custom"
    assert result.diagnostics["note"] == "plumbing-only"
    assert result.diagnostics["source"] == (3, 6)
    assert result.diagnostics["target"] == (4, 7)


def test_constant_reward_accepts_extended_reward_context() -> None:
    """ConstantReward tolerates the richer reward context unchanged."""
    reward_fn = ConstantReward(reward=0.75)
    context = RewardContext(
        step_index=1,
        measure_size=4,
        history=((3, 6),),
        step_delta=(1, 1),
        key_pitch_class=0,
        timed_chord_window=TimedChordWindow(
            chord_sequence=((0, 0), (3, 6)),
            bar_positions=(-1, 0),
            valid_mask=(False, True),
        ),
    )

    result = reward_fn((3, 6), (4, 7), context)

    assert result.reward == 0.75
    assert result.diagnostics["step_index"] == 1


def test_beat_role_diagnostic_reward_returns_reward_result() -> None:
    """BeatRoleDiagnosticReward returns the structured RewardResult type."""
    reward_fn = BeatRoleDiagnosticReward(reward=0.25)

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=0))

    assert isinstance(result, RewardResult)
    assert result.reward == 0.25


def test_beat_role_diagnostic_reward_reports_meter_derived_flags() -> None:
    """Beat-role diagnostics are derived from step index and measure size."""
    reward_fn = BeatRoleDiagnosticReward()
    context = RewardContext(
        step_index=3,
        measure_size=4,
        key_pitch_class=0,
        step_delta=(1, 1),
        history=((3, 6), (4, 7)),
        timed_chord_window=TimedChordWindow(
            chord_sequence=((0, 0), (3, 6), (4, 7)),
            bar_positions=(-1, 0, 3),
            valid_mask=(False, True, True),
        ),
    )

    result = reward_fn((3, 6), (4, 7), context)

    assert result.diagnostics["kind"] == "beat_role_diagnostic"
    assert result.diagnostics["measure_size"] == 4
    assert result.diagnostics["key_pitch_class"] == 0
    assert result.diagnostics["bar_position"] == 3
    assert not result.diagnostics["is_leading_beat"]
    assert not result.diagnostics["is_downbeat"]
    assert result.diagnostics["is_ending_beat"]
    assert result.diagnostics["step_delta"] == (1, 1)
    assert result.diagnostics["history_length"] == 2
    assert result.diagnostics["timed_window_valid_length"] == 2
    assert result.diagnostics["timed_window_last_bar_position"] == 3


def test_beat_role_diagnostic_reward_tolerates_missing_measure_size() -> None:
    """Beat-role diagnostics degrade gracefully when meter is not supplied."""
    reward_fn = BeatRoleDiagnosticReward()

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=2))

    assert result.diagnostics["bar_position"] is None
    assert result.diagnostics["is_leading_beat"] is None
    assert result.diagnostics["is_downbeat"]
    assert result.diagnostics["is_ending_beat"] is None
    assert result.diagnostics["timed_window_valid_length"] is None
    assert result.diagnostics["timed_window_last_bar_position"] is None
