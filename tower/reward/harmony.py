"""Rank-2 harmonic reward shaping terms."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from tower.reward.context import TowerRewardContext
from tower.reward.melody import (
    CONSONANT_INTERVALS_MOD_12,
    consonance_from_pitch_class,
    midi_to_octave,
)
from tower.reward.result import TowerRewardResult


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
    non_consonance_penalty: float = -0.5
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
            is_consonant = interval_pitch_class in CONSONANT_INTERVALS_MOD_12
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
                        sorted(CONSONANT_INTERVALS_MOD_12)
                    ),
                    "intervals": tuple(interval_rows),
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

    target_vertical_interval: int = 5
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


def _validate_rank_2_context(context: TowerRewardContext) -> None:
    if context.rank != 2:
        raise ValueError("rank-2 harmonic reward requires rank 2 context")


def _new_voice_index(context: TowerRewardContext) -> int:
    new_voice_index = context.new_facts.new_voice_index
    if new_voice_index is None:
        return 1
    return new_voice_index


def _validate_number(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
