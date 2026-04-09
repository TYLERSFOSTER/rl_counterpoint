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
