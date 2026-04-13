"""Temporary black-box reward implementations."""

from __future__ import annotations

from math import ceil
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


def midi_to_octave(midi_note_value: int) -> int:
    """Map a MIDI note value to its scientific-pitch octave number."""

    return -1 + midi_note_value // 12


def target_root_octave_deadline(
    *,
    initial_state: ChordState,
    target_root_octave: int,
    max_step_size: int,
    measure_size: int,
    average_step_fraction: float = 0.5,
) -> tuple[int, int]:
    """Return the rewardable deadline in steps and measures for the target-octave task."""
    if not initial_state:
        raise ValueError("initial_state must not be empty")
    if max_step_size < 1:
        raise ValueError("max_step_size must be at least 1")
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")
    if average_step_fraction <= 0.0:
        raise ValueError("average_step_fraction must be positive")

    initial_root_octave = midi_to_octave(initial_state[0])
    initial_semitone_distance = abs(initial_root_octave - target_root_octave) * 12
    expected_progress_per_step = max_step_size * average_step_fraction
    estimated_steps = ceil(initial_semitone_distance / expected_progress_per_step)
    deadline_measures = max(1, ceil(estimated_steps / measure_size))
    return deadline_measures * measure_size, deadline_measures


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


@dataclass(frozen=True)
class TargetRootOctaveReward:
    """Reward proximity to a target root octave plus terminal tail closeness."""

    distance_weight: float = 1.0
    terminal_window_reward: float = 10.0
    terminal_window_size: int = 3
    deadline_average_step_fraction: float = 0.5
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __call__(
        self,
        source: ChordState,
        target: ChordState,
        context: RewardContext,
    ) -> RewardResult:
        if not target:
            raise ValueError("target chord must not be empty")
        if context.target_root_octave is None:
            raise ValueError(
                "target_root_octave is required for TargetRootOctaveReward"
            )

        root_pitch = target[0]
        root_octave = midi_to_octave(root_pitch)
        octave_distance = abs(root_octave - context.target_root_octave)
        if self.terminal_window_size < 1:
            raise ValueError("terminal_window_size must be at least 1")

        deadline_step = None
        deadline_measures = None
        deadline_active = False
        if (
            context.measure_size is not None
            and context.max_step_size is not None
            and context.history
        ):
            deadline_step, deadline_measures = target_root_octave_deadline(
                initial_state=context.history[0],
                target_root_octave=context.target_root_octave,
                max_step_size=context.max_step_size,
                measure_size=context.measure_size,
                average_step_fraction=self.deadline_average_step_fraction,
            )
            deadline_active = context.step_index >= deadline_step

        distance_reward = (
            0.0 if deadline_active else self.distance_weight / (1 + octave_distance)
        )
        terminal_root_octaves = ()
        terminal_distances = ()
        terminal_closeness_scores = ()
        terminal_window_average = 0.0
        terminal_bonus = 0.0

        if context.is_final_step and not deadline_active:
            terminal_chords = (*context.history, target)[-self.terminal_window_size :]
            terminal_root_octaves = tuple(
                midi_to_octave(chord[0]) for chord in terminal_chords
            )
            terminal_distances = tuple(
                abs(root_octave_value - context.target_root_octave)
                for root_octave_value in terminal_root_octaves
            )
            terminal_closeness_scores = tuple(
                1 / (1 + distance) for distance in terminal_distances
            )
            terminal_window_average = sum(terminal_closeness_scores) / len(
                terminal_closeness_scores
            )
            terminal_bonus = self.terminal_window_reward * terminal_window_average

        terminal_match = (
            context.is_final_step and not deadline_active and terminal_window_average == 1.0
        )

        return RewardResult(
            reward=distance_reward + terminal_bonus,
            is_terminal_success=terminal_match,
            diagnostics={
                "kind": "target_root_octave",
                "source": source,
                "target": target,
                "step_index": context.step_index,
                "root_pitch": root_pitch,
                "root_octave": root_octave,
                "target_root_octave": context.target_root_octave,
                "octave_distance": octave_distance,
                "deadline_step": deadline_step,
                "deadline_measures": deadline_measures,
                "deadline_active": deadline_active,
                "distance_weight": self.distance_weight,
                "distance_reward": distance_reward,
                "is_final_step": context.is_final_step,
                "terminal_window_reward": self.terminal_window_reward,
                "terminal_window_size": self.terminal_window_size,
                "terminal_root_octaves": terminal_root_octaves,
                "terminal_distances": terminal_distances,
                "terminal_closeness_scores": terminal_closeness_scores,
                "terminal_window_average": terminal_window_average,
                "terminal_bonus": terminal_bonus,
                "terminal_match": terminal_match,
                **self.diagnostics,
            },
        )
