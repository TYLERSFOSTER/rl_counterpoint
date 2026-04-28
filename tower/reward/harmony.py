"""Rank-2 and rank-3 harmonic reward shaping terms."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from tower.reward.context import TowerRewardContext
from tower.reward.melody import (
    consonance_from_pitch_class,
    midi_to_octave,
)
from tower.reward.result import TowerRewardResult

RANK2_CONSONANT_INTERVALS_MOD_12 = frozenset({3, 4, 7, 8, 9})
RANK3_CONSONANT_INTERVALS_MOD_12 = frozenset({3, 4, 7, 8, 9})


@dataclass(frozen=True)
class Rank2TargetOctaveDistanceReward:
    """Reward the realized outer voice by inverse distance from target octave."""

    diagnostics_key: str = "rank2_target_octave_distance"

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_2_context(context)
        if context.target_root_octave is None:
            raise ValueError(
                "target_root_octave is required for Rank2TargetOctaveDistanceReward"
            )

        new_voice_index = _new_voice_index(context)
        target_pitch = context.target[new_voice_index]
        root_octave = midi_to_octave(target_pitch)
        octave_distance = abs(root_octave - context.target_root_octave)
        reward = 1.0 / (1.0 + octave_distance)

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank2_target_octave_distance",
                    "new_voice_index": new_voice_index,
                    "target_pitch": target_pitch,
                    "root_octave": root_octave,
                    "target_root_octave": context.target_root_octave,
                    "octave_distance": octave_distance,
                    "reward_formula": "1/(1+d)",
                }
            },
        )


@dataclass(frozen=True)
class Rank2VerticalConsonanceReward:
    """Reward consonant realized outer intervals and penalize non-consonance."""

    consonance_weight: float = 1.0
    non_consonance_penalty: float = 0.0
    diagnostics_key: str = "rank2_vertical_consonance"

    def __post_init__(self) -> None:
        _validate_number(self.consonance_weight, field_name="consonance_weight")
        _validate_number(
            self.non_consonance_penalty,
            field_name="non_consonance_penalty",
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_2_context(context)

        new_voice_index = _new_voice_index(context)
        new_voice_pitch = context.target[new_voice_index]
        interval_rows: list[dict[str, object]] = []
        reward = 0.0
        for other_index, other_pitch in enumerate(context.target):
            if other_index == new_voice_index:
                continue
            interval_pitch_class = abs(new_voice_pitch - other_pitch) % 12
            is_consonant = interval_pitch_class in RANK2_CONSONANT_INTERVALS_MOD_12
            consonance = consonance_from_pitch_class(interval_pitch_class)
            interval_reward = (
                float(self.consonance_weight) * consonance
                if is_consonant
                else float(self.non_consonance_penalty)
            )
            reward += interval_reward
            interval_rows.append(
                {
                    "other_voice_index": other_index,
                    "other_pitch": other_pitch,
                    "new_voice_pitch": new_voice_pitch,
                    "interval_pitch_class": interval_pitch_class,
                    "is_consonant": is_consonant,
                    "just_consonance": consonance,
                    "reward": interval_reward,
                }
            )

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank2_vertical_consonance",
                    "new_voice_index": new_voice_index,
                    "consonance_weight": float(self.consonance_weight),
                    "non_consonance_penalty": float(self.non_consonance_penalty),
                    "consonant_intervals_mod_12": tuple(
                        sorted(RANK2_CONSONANT_INTERVALS_MOD_12)
                    ),
                    "intervals": tuple(interval_rows),
                }
            },
        )


@dataclass(frozen=True)
class Rank2CadenceEndpointReward:
    """Give endpoint shaping for the child voice even when parent cadence fails."""

    weight: float = 1.0
    diagnostics_key: str = "rank2_cadence_endpoint"

    def __post_init__(self) -> None:
        _validate_number(self.weight, field_name="weight")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_2_context(context)
        if context.key_pitch_class is None:
            raise ValueError("key_pitch_class is required for Rank2CadenceEndpointReward")
        if not context.is_final_step:
            return TowerRewardResult(
                reward=0.0,
                diagnostics={
                    self.diagnostics_key: {
                        "kind": "rank2_cadence_endpoint",
                        "reason": "not_final_step",
                    }
                },
            )

        previous_outer_pitch_class = context.source[1] % 12
        final_outer_pitch_class = context.target[1] % 12
        dominant_third_pitch_class = (context.key_pitch_class + 11) % 12
        tonic_third_pitch_class = (context.key_pitch_class + 4) % 12
        previous_distance = _pitch_class_distance(
            previous_outer_pitch_class,
            dominant_third_pitch_class,
        )
        final_distance = _pitch_class_distance(
            final_outer_pitch_class,
            tonic_third_pitch_class,
        )
        reward = float(self.weight) * (
            (1.0 / (1.0 + previous_distance) + 1.0 / (1.0 + final_distance)) / 2.0
        )
        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank2_cadence_endpoint",
                    "previous_outer_pitch_class": previous_outer_pitch_class,
                    "final_outer_pitch_class": final_outer_pitch_class,
                    "dominant_third_pitch_class": dominant_third_pitch_class,
                    "tonic_third_pitch_class": tonic_third_pitch_class,
                    "previous_distance": previous_distance,
                    "final_distance": final_distance,
                    "weight": float(self.weight),
                    "reward_formula": "w * (1/(1+d_prev) + 1/(1+d_final)) / 2",
                }
            },
        )


@dataclass(frozen=True)
class Rank2SpacingControlReward:
    """Discourage upper-register saturation and collapsed vertical spacing."""

    upper_register_soft_ceiling: int = 80
    upper_register_penalty_weight: float = 0.05
    min_vertical_gap: int = 3
    spacing_reward: float = 0.1
    spacing_penalty: float = -0.1
    diagnostics_key: str = "rank2_spacing_control"

    def __post_init__(self) -> None:
        if not isinstance(self.upper_register_soft_ceiling, int):
            raise TypeError("upper_register_soft_ceiling must be an int")
        if self.upper_register_soft_ceiling < 0 or self.upper_register_soft_ceiling > 127:
            raise ValueError("upper_register_soft_ceiling must be in [0, 127]")
        _validate_number(
            self.upper_register_penalty_weight,
            field_name="upper_register_penalty_weight",
        )
        if not isinstance(self.min_vertical_gap, int):
            raise TypeError("min_vertical_gap must be an int")
        if self.min_vertical_gap < 1:
            raise ValueError("min_vertical_gap must be at least 1")
        _validate_number(self.spacing_reward, field_name="spacing_reward")
        _validate_number(self.spacing_penalty, field_name="spacing_penalty")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_2_context(context)

        new_voice_index = _new_voice_index(context)
        new_voice_pitch = context.target[new_voice_index]
        other_pitch = context.target[1 - new_voice_index]
        vertical_gap = new_voice_pitch - other_pitch
        excess_above_ceiling = max(
            0,
            new_voice_pitch - int(self.upper_register_soft_ceiling),
        )
        ceiling_penalty = -float(self.upper_register_penalty_weight) * float(
            excess_above_ceiling
        )
        spacing_component = (
            float(self.spacing_reward)
            if vertical_gap >= self.min_vertical_gap
            else float(self.spacing_penalty)
        )
        reward = ceiling_penalty + spacing_component

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank2_spacing_control",
                    "new_voice_index": new_voice_index,
                    "new_voice_pitch": new_voice_pitch,
                    "other_pitch": other_pitch,
                    "vertical_gap": vertical_gap,
                    "min_vertical_gap": self.min_vertical_gap,
                    "spacing_reward": float(self.spacing_reward),
                    "spacing_penalty": float(self.spacing_penalty),
                    "upper_register_soft_ceiling": int(
                        self.upper_register_soft_ceiling
                    ),
                    "upper_register_penalty_weight": float(
                        self.upper_register_penalty_weight
                    ),
                    "excess_above_ceiling": excess_above_ceiling,
                    "ceiling_penalty": ceiling_penalty,
                }
            },
        )


@dataclass(frozen=True)
class Rank2TargetVerticalIntervalReward:
    """Reward realized vertical gap by inverse distance from a target interval."""

    target_vertical_interval: int = 4
    interval_reward_weight: float = 1.0
    diagnostics_key: str = "rank2_target_vertical_interval"

    def __post_init__(self) -> None:
        if not isinstance(self.target_vertical_interval, int):
            raise TypeError("target_vertical_interval must be an int")
        if self.target_vertical_interval < 0:
            raise ValueError("target_vertical_interval must be non-negative")
        _validate_number(
            self.interval_reward_weight,
            field_name="interval_reward_weight",
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_2_context(context)

        new_voice_index = _new_voice_index(context)
        new_voice_pitch = context.target[new_voice_index]
        other_pitch = context.target[1 - new_voice_index]
        vertical_gap = new_voice_pitch - other_pitch
        interval_distance = abs(vertical_gap - self.target_vertical_interval)
        reward = float(self.interval_reward_weight) * (
            1.0 / (interval_distance + 1.0)
        )

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank2_target_vertical_interval",
                    "new_voice_index": new_voice_index,
                    "new_voice_pitch": new_voice_pitch,
                    "other_pitch": other_pitch,
                    "vertical_gap": vertical_gap,
                    "target_vertical_interval": self.target_vertical_interval,
                    "interval_distance": interval_distance,
                    "interval_reward_weight": float(self.interval_reward_weight),
                    "reward_formula": "w/(|gap-target|+1)",
                }
            },
        )


@dataclass(frozen=True)
class Rank3GlobalTriadConsonanceReward:
    """Score the realized three-voice sonority by pairwise consonance quality."""

    consonance_weight: float = 1.0
    non_consonance_penalty: float = 0.0
    diagnostics_key: str = "rank3_global_triad_consonance"

    def __post_init__(self) -> None:
        _validate_number(self.consonance_weight, field_name="consonance_weight")
        _validate_number(
            self.non_consonance_penalty,
            field_name="non_consonance_penalty",
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_3_context(context)

        interval_rows: list[dict[str, object]] = []
        reward = 0.0
        for lower_index, upper_index in ((0, 1), (1, 2), (0, 2)):
            interval_pitch_class = (context.target[upper_index] - context.target[lower_index]) % 12
            is_consonant = interval_pitch_class in RANK3_CONSONANT_INTERVALS_MOD_12
            consonance = consonance_from_pitch_class(interval_pitch_class)
            interval_reward = (
                float(self.consonance_weight) * consonance
                if is_consonant
                else float(self.non_consonance_penalty)
            )
            reward += interval_reward
            interval_rows.append(
                {
                    "lower_voice_index": lower_index,
                    "upper_voice_index": upper_index,
                    "interval_pitch_class": interval_pitch_class,
                    "is_consonant": is_consonant,
                    "just_consonance": consonance,
                    "reward": interval_reward,
                }
            )

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank3_global_triad_consonance",
                    "consonance_weight": float(self.consonance_weight),
                    "non_consonance_penalty": float(self.non_consonance_penalty),
                    "consonant_intervals_mod_12": tuple(
                        sorted(RANK3_CONSONANT_INTERVALS_MOD_12)
                    ),
                    "intervals": tuple(interval_rows),
                }
            },
        )


@dataclass(frozen=True)
class Rank3GlobalSpacingReward:
    """Reward uncluttered adjacent spacing and bounded outer span."""

    min_adjacent_gap: int = 3
    max_outer_span: int = 15
    adjacent_spacing_reward: float = 0.1
    adjacent_spacing_penalty: float = -0.1
    outer_span_reward: float = 0.1
    outer_span_penalty: float = -0.1
    diagnostics_key: str = "rank3_global_spacing"

    def __post_init__(self) -> None:
        if not isinstance(self.min_adjacent_gap, int):
            raise TypeError("min_adjacent_gap must be an int")
        if self.min_adjacent_gap < 1:
            raise ValueError("min_adjacent_gap must be at least 1")
        if not isinstance(self.max_outer_span, int):
            raise TypeError("max_outer_span must be an int")
        if self.max_outer_span < 1:
            raise ValueError("max_outer_span must be at least 1")
        _validate_number(self.adjacent_spacing_reward, field_name="adjacent_spacing_reward")
        _validate_number(self.adjacent_spacing_penalty, field_name="adjacent_spacing_penalty")
        _validate_number(self.outer_span_reward, field_name="outer_span_reward")
        _validate_number(self.outer_span_penalty, field_name="outer_span_penalty")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_3_context(context)

        lower_gap = context.target[1] - context.target[0]
        upper_gap = context.target[2] - context.target[1]
        outer_span = context.target[2] - context.target[0]
        lower_reward = (
            float(self.adjacent_spacing_reward)
            if lower_gap >= self.min_adjacent_gap
            else float(self.adjacent_spacing_penalty)
        )
        upper_reward = (
            float(self.adjacent_spacing_reward)
            if upper_gap >= self.min_adjacent_gap
            else float(self.adjacent_spacing_penalty)
        )
        outer_reward = (
            float(self.outer_span_reward)
            if outer_span <= self.max_outer_span
            else float(self.outer_span_penalty)
        )
        reward = lower_reward + upper_reward + outer_reward

        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank3_global_spacing",
                    "lower_gap": lower_gap,
                    "upper_gap": upper_gap,
                    "outer_span": outer_span,
                    "min_adjacent_gap": self.min_adjacent_gap,
                    "max_outer_span": self.max_outer_span,
                    "lower_reward": lower_reward,
                    "upper_reward": upper_reward,
                    "outer_reward": outer_reward,
                }
            },
        )


@dataclass(frozen=True)
class Rank3CadenceEndpointTriadReward:
    """Give final-step shaping toward the intended dominant and tonic triads."""

    weight: float = 1.0
    diagnostics_key: str = "rank3_cadence_endpoint_triad"

    def __post_init__(self) -> None:
        _validate_number(self.weight, field_name="weight")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        _validate_rank_3_context(context)
        if context.key_pitch_class is None:
            raise ValueError("key_pitch_class is required for Rank3CadenceEndpointTriadReward")
        if not context.is_final_step:
            return TowerRewardResult(
                reward=0.0,
                diagnostics={
                    self.diagnostics_key: {
                        "kind": "rank3_cadence_endpoint_triad",
                        "reason": "not_final_step",
                    }
                },
            )

        previous_expected = (
            (context.key_pitch_class + 7) % 12,
            (context.key_pitch_class + 2) % 12,
            (context.key_pitch_class + 11) % 12,
        )
        final_expected = (
            context.key_pitch_class % 12,
            (context.key_pitch_class + 7) % 12,
            (context.key_pitch_class + 4) % 12,
        )
        previous_pitch_classes = tuple(pitch % 12 for pitch in context.source)
        final_pitch_classes = tuple(pitch % 12 for pitch in context.target)
        previous_distances = tuple(
            _pitch_class_distance(actual, expected)
            for actual, expected in zip(previous_pitch_classes, previous_expected, strict=True)
        )
        final_distances = tuple(
            _pitch_class_distance(actual, expected)
            for actual, expected in zip(final_pitch_classes, final_expected, strict=True)
        )
        reward = float(self.weight) * (
            sum(1.0 / (distance + 1.0) for distance in previous_distances + final_distances)
            / 6.0
        )
        return TowerRewardResult(
            reward=reward,
            diagnostics={
                self.diagnostics_key: {
                    "kind": "rank3_cadence_endpoint_triad",
                    "previous_pitch_classes": previous_pitch_classes,
                    "final_pitch_classes": final_pitch_classes,
                    "previous_expected_pitch_classes": previous_expected,
                    "final_expected_pitch_classes": final_expected,
                    "previous_distances": previous_distances,
                    "final_distances": final_distances,
                    "weight": float(self.weight),
                    "reward_formula": "w * mean(1/(1+d_i)) over prev/final triad voices",
                }
            },
        )


def _validate_rank_2_context(context: TowerRewardContext) -> None:
    if context.rank != 2:
        raise ValueError("rank-2 harmonic reward requires rank 2 context")


def _validate_rank_3_context(context: TowerRewardContext) -> None:
    if context.rank != 3:
        raise ValueError("rank-3 harmonic reward requires rank 3 context")


def _new_voice_index(context: TowerRewardContext) -> int:
    new_voice_index = context.new_facts.new_voice_index
    if new_voice_index is None:
        return 1
    return new_voice_index


def _pitch_class_distance(a: int, b: int) -> int:
    clockwise = (a - b) % 12
    counterclockwise = (b - a) % 12
    return min(clockwise, counterclockwise)


def _validate_number(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
