"""Tests for the reward protocol objects.

These tests protect the reward interface that the environment will call. They
verify that reward context and result objects have stable defaults and can carry
the minimal episode/diagnostic data needed before the TC21M reward evaluator is
implemented.
"""

from __future__ import annotations

from rl_counterpoint.envs.observation import TimedChordWindow
from rl_counterpoint.graph.actions import StepDelta
from rl_counterpoint.reward.protocol import RewardContext, RewardResult


def test_reward_context_defaults() -> None:
    """A context can be created with only the current step index."""
    context = RewardContext(step_index=3)

    assert context.step_index == 3
    assert context.max_steps is None
    assert context.measure_size is None
    assert context.history == ()
    assert context.step_delta is None
    assert context.key_pitch_class is None
    assert context.timed_chord_window is None
    assert context.target_root_octave is None
    assert not context.is_final_step


def test_reward_context_accepts_history() -> None:
    """A context can carry prior chord states without referencing an env object."""
    history = ((3, 6), (4, 7))
    context = RewardContext(
        step_index=2,
        max_steps=8,
        measure_size=4,
        history=history,
    )

    assert context.step_index == 2
    assert context.max_steps == 8
    assert context.measure_size == 4
    assert context.history == history


def test_reward_context_accepts_step_delta_key_and_timed_window() -> None:
    """A context can carry the approved reward-boundary transition metadata."""
    step_delta: StepDelta = (1, -1)
    timed_window = TimedChordWindow(
        chord_sequence=((0, 0), (3, 6)),
        bar_positions=(-1, 0),
        valid_mask=(False, True),
    )
    context = RewardContext(
        step_index=2,
        measure_size=4,
        history=((3, 6),),
        step_delta=step_delta,
        key_pitch_class=0,
        timed_chord_window=timed_window,
    )

    assert context.step_delta == step_delta
    assert context.key_pitch_class == 0
    assert context.timed_chord_window == timed_window


def test_reward_context_accepts_target_octave_and_final_step() -> None:
    """A context can carry target-octave and final-step reward metadata."""
    context = RewardContext(
        step_index=7,
        target_root_octave=4,
        is_final_step=True,
    )

    assert context.target_root_octave == 4
    assert context.is_final_step


def test_reward_result_defaults() -> None:
    """A reward result can be a scalar reward with default status flags."""
    result = RewardResult(reward=1.5)

    assert result.reward == 1.5
    assert not result.hard_violation
    assert not result.is_terminal_success
    assert result.diagnostics == {}


def test_reward_result_accepts_diagnostics_and_flags() -> None:
    """A reward result can carry status flags and diagnostic payloads."""
    diagnostics = {"kind": "example", "voice": "soprano"}
    result = RewardResult(
        reward=-2.0,
        hard_violation=True,
        is_terminal_success=False,
        diagnostics=diagnostics,
    )

    assert result.reward == -2.0
    assert result.hard_violation
    assert not result.is_terminal_success
    assert result.diagnostics == diagnostics
