"""Rank-1 melodic reward shaping terms."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from tower.reward.context import TowerRewardContext
from tower.reward.result import TowerRewardResult

JUST_INTERVAL_RATIOS_BY_PITCH_CLASS: dict[int, tuple[int, int]] = {
    0: (1, 1),
    1: (16, 15),
    2: (9, 8),
    3: (6, 5),
    4: (5, 4),
    5: (4, 3),
    6: (45, 32),
    7: (3, 2),
    8: (8, 5),
    9: (5, 3),
    10: (9, 5),
    11: (15, 8),
}

MAJOR_SCALE_INTERVALS_MOD_12 = frozenset({0, 2, 4, 5, 7, 9, 11})
CONSONANT_INTERVALS_MOD_12 = frozenset({0, 3, 4, 5, 7, 8, 9})


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
class GoalOctaveDirectionReward:
    """Reward rank-1 motion that reduces octave distance to the goal."""

    weight: float = 0.5
    diagnostics_key: str = "goal_octave_direction"

    def __post_init__(self) -> None:
        _validate_number(self.weight, field_name="weight")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_1_context(context)
        if context.target_root_octave is None:
            raise ValueError(
                "target_root_octave is required for GoalOctaveDirectionReward"
            )

        source_pitch = context.source[0]
        target_pitch = context.target[0]
        source_octave = midi_to_octave(source_pitch)
        target_octave = midi_to_octave(target_pitch)
        source_distance = abs(source_octave - context.target_root_octave)
        target_distance = abs(target_octave - context.target_root_octave)
        distance_delta = source_distance - target_distance
        if distance_delta > 0:
            reward = float(self.weight)
            direction = "toward_goal_octave"
        elif distance_delta < 0:
            reward = -float(self.weight)
            direction = "away_from_goal_octave"
        else:
            reward = 0.0
            direction = "no_octave_distance_change"

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "goal_octave_direction",
                    "source_pitch": source_pitch,
                    "target_pitch": target_pitch,
                    "source_octave": source_octave,
                    "target_octave": target_octave,
                    "target_root_octave": context.target_root_octave,
                    "source_octave_distance": source_distance,
                    "target_octave_distance": target_distance,
                    "distance_delta": distance_delta,
                    "direction": direction,
                    "weight": float(self.weight),
                    "reward_formula": "+w if d_target<d_source; -w if d_target>d_source; 0 otherwise",
                }
            },
        )


@dataclass(frozen=True)
class BeatClassPitchReward:
    """Reward rank-1 pitches according to metrical role and key relation."""

    measure_start_tonic_reward: float = 1.0
    onbeat_scale_degree_reward: float = 1.0
    offbeat_consonance_weight: float = 1.0
    onbeat_non_scale_penalty: float = -2.0
    offbeat_non_consonance_penalty: float = -2.0
    diagnostics_key: str = "beat_class_pitch"

    def __post_init__(self) -> None:
        _validate_number(
            self.measure_start_tonic_reward,
            field_name="measure_start_tonic_reward",
        )
        _validate_number(
            self.onbeat_scale_degree_reward,
            field_name="onbeat_scale_degree_reward",
        )
        _validate_number(
            self.offbeat_consonance_weight,
            field_name="offbeat_consonance_weight",
        )
        _validate_number(
            self.onbeat_non_scale_penalty,
            field_name="onbeat_non_scale_penalty",
        )
        _validate_number(
            self.offbeat_non_consonance_penalty,
            field_name="offbeat_non_consonance_penalty",
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_1_context(context)
        if context.key_pitch_class is None:
            raise ValueError("key_pitch_class is required for BeatClassPitchReward")
        if context.measure_size is None:
            raise ValueError("measure_size is required for BeatClassPitchReward")

        target_pitch = context.target[0]
        pitch_class = target_pitch % 12
        relative_pitch_class = (pitch_class - context.key_pitch_class) % 12
        landing_step_index = context.step_index + 1
        bar_position = landing_step_index % context.measure_size
        measure_index = landing_step_index // context.measure_size
        is_measure_start = bar_position == 0
        is_onbeat = bar_position % 2 == 0
        is_offbeat = bar_position % 2 == 1
        is_tonic = relative_pitch_class == 0
        is_scale_degree = relative_pitch_class in MAJOR_SCALE_INTERVALS_MOD_12
        is_consonant = relative_pitch_class in CONSONANT_INTERVALS_MOD_12
        consonance = consonance_from_pitch_class(relative_pitch_class)

        measure_start_reward = (
            float(self.measure_start_tonic_reward)
            if is_measure_start and is_tonic
            else 0.0
        )
        onbeat_reward = (
            float(self.onbeat_scale_degree_reward)
            if is_onbeat and is_scale_degree
            else 0.0
        )
        onbeat_penalty = (
            float(self.onbeat_non_scale_penalty)
            if is_onbeat and not is_scale_degree
            else 0.0
        )
        offbeat_reward = (
            float(self.offbeat_consonance_weight) * consonance if is_offbeat else 0.0
        )
        offbeat_penalty = (
            float(self.offbeat_non_consonance_penalty)
            if is_offbeat and not is_consonant
            else 0.0
        )

        return TowerRewardResult(
            reward=(
                measure_start_reward
                + onbeat_reward
                + onbeat_penalty
                + offbeat_reward
                + offbeat_penalty
            ),
            diagnostics={
                self.diagnostics_key: {
                    "kind": "beat_class_pitch",
                    "target_pitch": target_pitch,
                    "pitch_class": pitch_class,
                    "key_pitch_class": context.key_pitch_class,
                    "relative_pitch_class": relative_pitch_class,
                    "source_step_index": context.step_index,
                    "landing_step_index": landing_step_index,
                    "measure_size": context.measure_size,
                    "measure_index": measure_index,
                    "bar_position": bar_position,
                    "is_measure_start": is_measure_start,
                    "is_onbeat": is_onbeat,
                    "is_offbeat": is_offbeat,
                    "is_tonic": is_tonic,
                    "is_scale_degree": is_scale_degree,
                    "is_consonant": is_consonant,
                    "scale_intervals_mod_12": tuple(sorted(MAJOR_SCALE_INTERVALS_MOD_12)),
                    "consonant_intervals_mod_12": tuple(
                        sorted(CONSONANT_INTERVALS_MOD_12)
                    ),
                    "just_consonance": consonance,
                    "measure_start_tonic_reward": measure_start_reward,
                    "onbeat_scale_degree_reward": onbeat_reward,
                    "onbeat_non_scale_penalty": onbeat_penalty,
                    "offbeat_consonance_reward": offbeat_reward,
                    "offbeat_non_consonance_penalty": offbeat_penalty,
                    "beat_class_timing": "target_landing_step_index",
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


@dataclass(frozen=True)
class StepSizeBinBalanceReward:
    """Reward a targeted recent mix of small and large rank-1 steps."""

    small_step_threshold: int = 3
    target_small_rate: float = 0.3
    weight: float = 1.0
    min_interval_count: int = 2
    diagnostics_key: str = "step_size_bin_balance"

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.small_step_threshold,
            field_name="small_step_threshold",
        )
        _validate_rate(self.target_small_rate, field_name="target_small_rate")
        _validate_number(self.weight, field_name="weight")
        _validate_positive_int(
            self.min_interval_count,
            field_name="min_interval_count",
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_1_context(context)
        intervals = _recent_rank_1_intervals_including_action(context)
        absolute_step_sizes = tuple(abs(interval) for interval in intervals)
        small_count = sum(
            1 for step_size in absolute_step_sizes if step_size <= self.small_step_threshold
        )
        large_count = len(absolute_step_sizes) - small_count
        total_count = len(absolute_step_sizes)
        diagnostics: dict[str, object] = {
            "kind": "step_size_bin_balance",
            "small_step_threshold": self.small_step_threshold,
            "target_small_rate": self.target_small_rate,
            "target_large_rate": 1.0 - self.target_small_rate,
            "weight": self.weight,
            "min_interval_count": self.min_interval_count,
            "intervals": intervals,
            "absolute_step_sizes": absolute_step_sizes,
            "small_count": small_count,
            "large_count": large_count,
            "total_count": total_count,
            "max_step_size": context.max_step_size,
        }

        if total_count < self.min_interval_count:
            diagnostics.update(
                {
                    "observed_small_rate": None,
                    "observed_large_rate": None,
                    "balance_score": 0.0,
                    "reason": "insufficient_intervals",
                }
            )
            return TowerRewardResult(
                reward=0.0,
                diagnostics={self.diagnostics_key: diagnostics},
            )

        observed_small_rate = small_count / total_count
        observed_large_rate = large_count / total_count
        if observed_small_rate <= self.target_small_rate:
            balance_score = observed_small_rate / self.target_small_rate
        else:
            balance_score = observed_large_rate / (1.0 - self.target_small_rate)
        signed_balance_score = 2.0 * balance_score - 1.0
        diagnostics.update(
            {
                "observed_small_rate": observed_small_rate,
                "observed_large_rate": observed_large_rate,
                "balance_score": balance_score,
                "signed_balance_score": signed_balance_score,
                "reason": "target_matched" if balance_score == 1.0 else "off_target",
            }
        )
        return TowerRewardResult(
            reward=float(self.weight) * signed_balance_score,
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


def _recent_rank_1_intervals_including_action(
    context: TowerRewardContext,
) -> tuple[int, ...]:
    pitches = _valid_rank_1_pitches(context)
    historical_intervals = tuple(
        current - previous for previous, current in zip(pitches, pitches[1:])
    )
    return historical_intervals + (context.action[0],)


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


def _validate_rate(value: float, *, field_name: str) -> None:
    _validate_number(value, field_name=field_name)
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def midi_to_octave(pitch: int) -> int:
    """Return scientific pitch octave for a MIDI pitch number."""
    if isinstance(pitch, bool) or not isinstance(pitch, int):
        raise TypeError("pitch must be an integer")
    if pitch < 0 or pitch > 127:
        raise ValueError("pitch must be in [0, 127]")
    return pitch // 12 - 1


def consonance_from_pitch_class(interval_pitch_class: int) -> float:
    """Return the just-ratio consonance score for one pitch class."""
    if isinstance(interval_pitch_class, bool) or not isinstance(interval_pitch_class, int):
        raise TypeError("interval_pitch_class must be an integer")
    if interval_pitch_class < 0 or interval_pitch_class > 11:
        raise ValueError("interval_pitch_class must be in [0, 11]")

    numerator, denominator = JUST_INTERVAL_RATIOS_BY_PITCH_CLASS[interval_pitch_class]
    return 1.0 / (numerator + denominator)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)
