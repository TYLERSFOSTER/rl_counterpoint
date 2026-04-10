"""Temporary black-box reward implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from rl_counterpoint.envs.observation import (
    PAD_METRICAL_POSITION,
    is_downbeat,
    is_ending_beat,
    is_leading_beat,
)
from rl_counterpoint.graph.state_space import ChordState
from rl_counterpoint.music.consonance import consonance_from_pitch_class
from rl_counterpoint.music.intervals import pitch_class_interval
from rl_counterpoint.reward.protocol import RewardContext, RewardResult


@dataclass(frozen=True)
class ConstantReward:
    """Return a fixed reward for every transition.

    This is a plumbing reward, not a music-theory reward. It exists so the
    environment and learner can depend on the reward protocol before the TC21M
    evaluator is formalized.
    """

    reward: float = 0.0
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __call__(
        self,
        source: ChordState,
        target: ChordState,
        context: RewardContext,
    ) -> RewardResult:
        return RewardResult(
            reward=self.reward,
            diagnostics={
                "kind": "constant",
                "source": source,
                "target": target,
                "step_index": context.step_index,
                **self.diagnostics,
            },
        )


@dataclass(frozen=True)
class BeatRoleDiagnosticReward:
    """Return a fixed reward while exposing beat-role diagnostics from context."""

    reward: float = 0.0
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __call__(
        self,
        source: ChordState,
        target: ChordState,
        context: RewardContext,
    ) -> RewardResult:
        bar_pos = None
        if context.measure_size is not None:
            bar_pos = context.step_index % context.measure_size

        timed_window_valid_length = None
        timed_window_last_bar_position = None
        if context.timed_chord_window is not None:
            timed_window_valid_length = sum(context.timed_chord_window.valid_mask)
            valid_positions = [
                bar_position
                for bar_position, is_valid in zip(
                    context.timed_chord_window.bar_positions,
                    context.timed_chord_window.valid_mask,
                    strict=True,
                )
                if is_valid
            ]
            if valid_positions:
                timed_window_last_bar_position = valid_positions[-1]
            else:
                timed_window_last_bar_position = PAD_METRICAL_POSITION

        return RewardResult(
            reward=self.reward,
            diagnostics={
                "kind": "beat_role_diagnostic",
                "source": source,
                "target": target,
                "step_index": context.step_index,
                "measure_size": context.measure_size,
                "key_pitch_class": context.key_pitch_class,
                "bar_position": bar_pos,
                "is_leading_beat": (
                    None
                    if context.measure_size is None
                    else is_leading_beat(
                        step_index=context.step_index,
                        measure_size=context.measure_size,
                    )
                ),
                "is_downbeat": is_downbeat(step_index=context.step_index),
                "is_ending_beat": (
                    None
                    if context.measure_size is None
                    else is_ending_beat(
                        step_index=context.step_index,
                        measure_size=context.measure_size,
                    )
                ),
                "step_delta": context.step_delta,
                "history_length": len(context.history),
                "timed_window_valid_length": timed_window_valid_length,
                "timed_window_last_bar_position": timed_window_last_bar_position,
                **self.diagnostics,
            },
        )


@dataclass(frozen=True)
class StaticConsonanceReward:
    """Score the target chord by adjacent-interval and key-relative consonance."""

    adjacent_interval_weight: float = 1.0
    key_relative_weight: float = 1.0
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __call__(
        self,
        source: ChordState,
        target: ChordState,
        context: RewardContext,
    ) -> RewardResult:
        if not target:
            raise ValueError("target chord must not be empty")
        if context.key_pitch_class is None:
            raise ValueError("key_pitch_class is required for StaticConsonanceReward")

        adjacent_interval_pitch_classes = tuple(
            pitch_class_interval(lower_pitch, upper_pitch)
            for lower_pitch, upper_pitch in zip(target[:-1], target[1:], strict=True)
        )
        adjacent_interval_consonances = tuple(
            consonance_from_pitch_class(interval_pitch_class)
            for interval_pitch_class in adjacent_interval_pitch_classes
        )
        adjacent_interval_sum = sum(adjacent_interval_consonances)

        key_relative_pitch_classes = tuple(
            (pitch - context.key_pitch_class) % 12
            for pitch in target
        )
        key_relative_consonances = tuple(
            consonance_from_pitch_class(interval_pitch_class)
            for interval_pitch_class in key_relative_pitch_classes
        )
        key_relative_sum = sum(key_relative_consonances)

        reward = (
            self.adjacent_interval_weight * adjacent_interval_sum
            + self.key_relative_weight * key_relative_sum
        )

        return RewardResult(
            reward=reward,
            diagnostics={
                "kind": "static_consonance",
                "source": source,
                "target": target,
                "step_index": context.step_index,
                "key_pitch_class": context.key_pitch_class,
                "adjacent_interval_weight": self.adjacent_interval_weight,
                "key_relative_weight": self.key_relative_weight,
                "adjacent_interval_pitch_classes": adjacent_interval_pitch_classes,
                "adjacent_interval_consonances": adjacent_interval_consonances,
                "adjacent_interval_sum": adjacent_interval_sum,
                "key_relative_pitch_classes": key_relative_pitch_classes,
                "key_relative_consonances": key_relative_consonances,
                "key_relative_sum": key_relative_sum,
                **self.diagnostics,
            },
        )


@dataclass(frozen=True)
class StrongBeatConsonanceReward:
    """Reward static consonance on strong beats only, using even step parity."""

    adjacent_interval_weight: float = 1.0
    key_relative_weight: float = 1.0
    strong_beat_weight: float = 1.0
    weak_beat_weight: float = 0.0
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __call__(
        self,
        source: ChordState,
        target: ChordState,
        context: RewardContext,
    ) -> RewardResult:
        static_result = StaticConsonanceReward(
            adjacent_interval_weight=self.adjacent_interval_weight,
            key_relative_weight=self.key_relative_weight,
        )(source, target, context)
        strong_beat = is_downbeat(step_index=context.step_index)
        applied_beat_weight = (
            self.strong_beat_weight if strong_beat else self.weak_beat_weight
        )

        return RewardResult(
            reward=applied_beat_weight * static_result.reward,
            diagnostics={
                "kind": "strong_beat_consonance",
                "source": source,
                "target": target,
                "step_index": context.step_index,
                "is_strong_beat": strong_beat,
                "applied_beat_weight": applied_beat_weight,
                "base_static_consonance_reward": static_result.reward,
                "adjacent_interval_weight": self.adjacent_interval_weight,
                "key_relative_weight": self.key_relative_weight,
                "strong_beat_weight": self.strong_beat_weight,
                "weak_beat_weight": self.weak_beat_weight,
                "static_consonance_diagnostics": static_result.diagnostics,
                **self.diagnostics,
            },
        )
