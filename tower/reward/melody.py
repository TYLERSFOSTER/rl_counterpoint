"""Rank-1 melodic reward shaping terms."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from tower.reward.context import TowerRewardContext
from tower.reward.result import TowerRewardResult


@dataclass(frozen=True)
class TargetOctaveDistanceReward:
    """Reward rank-1 states by inverse distance from the target octave."""

    diagnostics_key: str = "target_octave_distance"

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_1_context(context)
        if context.target_root_octave is None:
            raise ValueError(
                "target_root_octave is required for TargetOctaveDistanceReward"
            )

        target_pitch = context.target[0]
        root_octave = midi_to_octave(target_pitch)
        octave_distance = abs(root_octave - context.target_root_octave)
        reward = 1.0 / (1.0 + octave_distance)

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "target_octave_distance",
                    "target_pitch": target_pitch,
                    "root_octave": root_octave,
                    "target_root_octave": context.target_root_octave,
                    "octave_distance": octave_distance,
                    "reward_formula": "1/(1+d)",
                }
            },
        )


@dataclass(frozen=True)
class RecentMelodicRangePenalty:
    """Penalize rank-1 melodies whose recent valid window spans too widely."""

    max_recent_range: int = 12
    penalty: float = -1.0
    diagnostics_key: str = "recent_melodic_range"

    def __post_init__(self) -> None:
        _validate_non_negative_int(
            self.max_recent_range,
            field_name="max_recent_range",
        )
        _validate_number(self.penalty, field_name="penalty")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_1_context(context)
        pitches = _valid_rank_1_pitches(context)
        diagnostics: dict[str, object] = {
            "kind": "recent_melodic_range_penalty",
            "valid_pitch_count": len(pitches),
            "max_recent_range": self.max_recent_range,
        }

        if len(pitches) < 2:
            diagnostics.update(
                {
                    "observed_range": None,
                    "penalty_applied": False,
                    "reason": "insufficient_valid_history",
                }
            )
            return TowerRewardResult(
                reward=0.0,
                diagnostics={self.diagnostics_key: diagnostics},
            )

        observed_range = max(pitches) - min(pitches)
        penalty_applied = observed_range > self.max_recent_range
        diagnostics.update(
            {
                "observed_range": observed_range,
                "penalty_applied": penalty_applied,
                "reason": "range_exceeded" if penalty_applied else "within_range",
            }
        )

        return TowerRewardResult(
            reward=float(self.penalty) if penalty_applied else 0.0,
            diagnostics={self.diagnostics_key: diagnostics},
        )


@dataclass(frozen=True)
class LargeLeapRecoveryTerm:
    """Reward immediate stepwise recovery after a large rank-1 melodic leap."""

    large_leap_threshold: int = 6
    recovery_step_threshold: int = 3
    recovery_reward: float = 0.5
    failure_penalty: float = -0.5
    diagnostics_key: str = "large_leap_recovery"

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.large_leap_threshold,
            field_name="large_leap_threshold",
        )
        _validate_positive_int(
            self.recovery_step_threshold,
            field_name="recovery_step_threshold",
        )
        _validate_number(self.recovery_reward, field_name="recovery_reward")
        _validate_number(self.failure_penalty, field_name="failure_penalty")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_1_context(context)
        pitches = _valid_rank_1_pitches(context)
        current_action = context.action[0]
        diagnostics: dict[str, object] = {
            "kind": "large_leap_recovery",
            "valid_pitch_count": len(pitches),
            "current_action": current_action,
            "large_leap_threshold": self.large_leap_threshold,
            "recovery_step_threshold": self.recovery_step_threshold,
        }

        if len(pitches) < 2:
            diagnostics.update(
                {
                    "previous_interval": None,
                    "triggered": False,
                    "opposite_direction": False,
                    "small_step": False,
                    "success": False,
                    "reason": "insufficient_valid_history",
                }
            )
            return TowerRewardResult(
                reward=0.0,
                diagnostics={self.diagnostics_key: diagnostics},
            )

        previous_interval = pitches[-1] - pitches[-2]
        triggered = abs(previous_interval) >= self.large_leap_threshold
        current_is_nonzero = current_action != 0
        opposite_direction = (
            current_is_nonzero and _sign(current_action) == -_sign(previous_interval)
        )
        small_step = abs(current_action) <= self.recovery_step_threshold
        success = triggered and opposite_direction and small_step

        diagnostics.update(
            {
                "previous_interval": previous_interval,
                "triggered": triggered,
                "opposite_direction": opposite_direction,
                "small_step": small_step,
                "success": success,
            }
        )

        if not triggered:
            diagnostics["reason"] = "no_large_leap"
            return TowerRewardResult(
                reward=0.0,
                diagnostics={self.diagnostics_key: diagnostics},
            )

        diagnostics["reason"] = "recovered" if success else "failed_recovery"
        return TowerRewardResult(
            reward=float(self.recovery_reward if success else self.failure_penalty),
            diagnostics={self.diagnostics_key: diagnostics},
        )


def _validate_rank_1_context(context: TowerRewardContext) -> None:
    if not isinstance(context, TowerRewardContext):
        raise TypeError("context must be a TowerRewardContext")
    if context.rank != 1:
        raise ValueError("melodic reward terms require rank 1 context")


def _valid_rank_1_pitches(context: TowerRewardContext) -> tuple[int, ...]:
    return tuple(
        state[0]
        for state, is_valid in zip(
            context.window.states,
            context.window.valid_mask,
            strict=True,
        )
        if is_valid
    )


def _validate_non_negative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")


def _validate_number(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")


def midi_to_octave(pitch: int) -> int:
    """Return scientific pitch octave for a MIDI pitch number."""
    if isinstance(pitch, bool) or not isinstance(pitch, int):
        raise TypeError("pitch must be an integer")
    if pitch < 0 or pitch > 127:
        raise ValueError("pitch must be in [0, 127]")
    return pitch // 12 - 1


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)
