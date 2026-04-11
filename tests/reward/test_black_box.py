"""Tests for temporary black-box reward implementations.

These tests cover placeholder rewards used before the TC21M evaluator exists.
They verify that the placeholder obeys the reward protocol and exposes enough
diagnostic information for environment plumbing and smoke tests.
"""

from __future__ import annotations

import pytest

from rl_counterpoint.envs.observation import TimedChordWindow
from rl_counterpoint.reward.black_box import (
    BeatRoleDiagnosticReward,
    ConstantReward,
    StrongBeatConsonanceReward,
    StaticConsonanceReward,
    TargetRootOctaveReward,
    consonance_from_pitch_class,
    midi_to_octave,
)
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
        target_root_octave=4,
        is_final_step=False,
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


def test_static_consonance_reward_returns_reward_result() -> None:
    """StaticConsonanceReward returns a structured RewardResult."""
    reward_fn = StaticConsonanceReward()

    result = reward_fn(
        (3, 6),
        (60, 64, 67),
        RewardContext(step_index=0, key_pitch_class=0),
    )

    assert isinstance(result, RewardResult)


def test_static_consonance_reward_scores_target_state_terms_separately() -> None:
    """StaticConsonanceReward reports adjacent and key-relative term breakdowns."""
    reward_fn = StaticConsonanceReward(
        adjacent_interval_weight=2.0,
        key_relative_weight=3.0,
    )

    result = reward_fn(
        (3, 6),
        (60, 64, 67),
        RewardContext(step_index=0, key_pitch_class=0),
    )

    expected_adjacent_sum = (
        consonance_from_pitch_class(4) + consonance_from_pitch_class(3)
    )
    expected_key_relative_sum = (
        consonance_from_pitch_class(0)
        + consonance_from_pitch_class(4)
        + consonance_from_pitch_class(7)
    )
    expected_reward = 2.0 * expected_adjacent_sum + 3.0 * expected_key_relative_sum

    assert result.reward == pytest.approx(expected_reward)
    assert result.diagnostics["kind"] == "static_consonance"
    assert result.diagnostics["adjacent_interval_pitch_classes"] == (4, 3)
    assert result.diagnostics["key_relative_pitch_classes"] == (0, 4, 7)
    assert result.diagnostics["adjacent_interval_sum"] == pytest.approx(
        expected_adjacent_sum
    )
    assert result.diagnostics["key_relative_sum"] == pytest.approx(
        expected_key_relative_sum
    )


def test_static_consonance_reward_requires_key_pitch_class() -> None:
    """StaticConsonanceReward needs the approved key encoding in context."""
    reward_fn = StaticConsonanceReward()

    with pytest.raises(
        ValueError,
        match="key_pitch_class is required for StaticConsonanceReward",
    ):
        reward_fn((3, 6), (60, 64, 67), RewardContext(step_index=0))


def test_strong_beat_consonance_reward_applies_static_reward_on_strong_beats() -> None:
    """StrongBeatConsonanceReward passes through static consonance on even beats."""
    reward_fn = StrongBeatConsonanceReward(
        adjacent_interval_weight=2.0,
        key_relative_weight=3.0,
    )

    result = reward_fn(
        (3, 6),
        (60, 64, 67),
        RewardContext(step_index=2, key_pitch_class=0),
    )

    expected_adjacent_sum = (
        consonance_from_pitch_class(4) + consonance_from_pitch_class(3)
    )
    expected_key_relative_sum = (
        consonance_from_pitch_class(0)
        + consonance_from_pitch_class(4)
        + consonance_from_pitch_class(7)
    )
    expected_static_reward = 2.0 * expected_adjacent_sum + 3.0 * expected_key_relative_sum

    assert result.reward == pytest.approx(expected_static_reward)
    assert result.diagnostics["kind"] == "strong_beat_consonance"
    assert result.diagnostics["is_strong_beat"]
    assert result.diagnostics["applied_beat_weight"] == pytest.approx(1.0)
    assert result.diagnostics["base_static_consonance_reward"] == pytest.approx(
        expected_static_reward
    )


def test_strong_beat_consonance_reward_zeroes_reward_on_weak_beats_by_default() -> None:
    """StrongBeatConsonanceReward defaults to no consonance reward on weak beats."""
    reward_fn = StrongBeatConsonanceReward()

    result = reward_fn(
        (3, 6),
        (60, 64, 67),
        RewardContext(step_index=1, key_pitch_class=0),
    )

    assert result.reward == pytest.approx(0.0)
    assert not result.diagnostics["is_strong_beat"]
    assert result.diagnostics["applied_beat_weight"] == pytest.approx(0.0)
    assert result.diagnostics["base_static_consonance_reward"] > 0.0


def test_midi_to_octave_uses_standard_midi_scientific_pitch_mapping() -> None:
    """Octave numbering increments at C under the approved MIDI convention."""

    assert midi_to_octave(0) == -1
    assert midi_to_octave(12) == 0
    assert midi_to_octave(60) == 4
    assert midi_to_octave(69) == 4
    assert midi_to_octave(127) == 9


def test_target_root_octave_reward_uses_inverse_octave_distance() -> None:
    """Per-step shaping reward is inverse distance from target root octave."""
    reward_fn = TargetRootOctaveReward(distance_weight=2.0, terminal_match_reward=9.0)

    result = reward_fn(
        (48, 55, 60),
        (60, 64, 67),
        RewardContext(step_index=3, target_root_octave=4),
    )

    assert result.reward == pytest.approx(2.0)
    assert not result.is_terminal_success
    assert result.diagnostics["root_octave"] == 4
    assert result.diagnostics["octave_distance"] == 0
    assert result.diagnostics["distance_reward"] == pytest.approx(2.0)
    assert result.diagnostics["terminal_bonus"] == pytest.approx(0.0)


def test_target_root_octave_reward_decreases_with_distance() -> None:
    """Farther target-octave distances receive smaller shaping reward."""
    reward_fn = TargetRootOctaveReward(distance_weight=3.0)

    result = reward_fn(
        (48, 55, 60),
        (36, 40, 43),
        RewardContext(step_index=1, target_root_octave=4),
    )

    assert result.reward == pytest.approx(3.0 / 3.0)
    assert result.diagnostics["root_octave"] == 2
    assert result.diagnostics["octave_distance"] == 2


def test_target_root_octave_reward_adds_large_bonus_for_final_exact_match() -> None:
    """Exact final-step arrival earns the configured terminal success bonus."""
    reward_fn = TargetRootOctaveReward(distance_weight=1.5, terminal_match_reward=8.0)

    result = reward_fn(
        (48, 55, 60),
        (60, 64, 67),
        RewardContext(step_index=7, target_root_octave=4, is_final_step=True),
    )

    assert result.reward == pytest.approx(9.5)
    assert result.is_terminal_success
    assert result.diagnostics["terminal_match"]
    assert result.diagnostics["terminal_bonus"] == pytest.approx(8.0)


def test_target_root_octave_reward_requires_target_octave() -> None:
    """TargetRootOctaveReward needs the approved target-octave context field."""
    reward_fn = TargetRootOctaveReward()

    with pytest.raises(
        ValueError,
        match="target_root_octave is required for TargetRootOctaveReward",
    ):
        reward_fn((48, 55, 60), (60, 64, 67), RewardContext(step_index=0))
