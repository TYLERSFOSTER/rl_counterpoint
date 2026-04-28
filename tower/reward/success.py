"""Terminal success predicates for tower rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tower.graph.projection import project_action, project_state, project_window
from tower.reward.context import TowerRewardContext
from tower.reward.melody import midi_to_octave
from tower.state_action import TowerState


@dataclass(frozen=True)
class SuccessResult:
    """Structured terminal-success predicate output."""

    success: bool
    diagnostics: Mapping[str, object]


def rank1_projected_cadence_success(
    context: TowerRewardContext,
) -> SuccessResult:
    """Return whether a rank-1 window realizes terminal V-I root motion."""
    if context.rank != 1:
        raise ValueError("rank1_projected_cadence_success requires rank 1 context")

    diagnostics: dict[str, object] = {
        "rank": context.rank,
        "kind": "rank1_projected_cadence_success",
    }

    if context.key_pitch_class is None:
        return _failure(diagnostics, reason="missing_key_pitch_class")
    if context.measure_size is None:
        return _failure(diagnostics, reason="missing_measure_size")
    if not context.is_final_step:
        return _failure(diagnostics, reason="not_final_step")

    final_bar_position = context.window.bar_positions[-1]
    if final_bar_position != context.measure_size - 1:
        return _failure(
            diagnostics,
            reason="wrong_metrical_position",
            final_bar_position=final_bar_position,
            expected_bar_position=context.measure_size - 1,
        )

    dominant_pitch_class = (context.key_pitch_class + 7) % 12
    tonic_pitch_class = context.key_pitch_class % 12
    previous_pitch = context.source[0]
    final_pitch = context.target[0]
    previous_pitch_class = previous_pitch % 12
    final_pitch_class = final_pitch % 12

    diagnostics.update(
        {
            "previous_pitch_class": previous_pitch_class,
            "final_pitch_class": final_pitch_class,
            "dominant_pitch_class": dominant_pitch_class,
            "tonic_pitch_class": tonic_pitch_class,
        }
    )

    if (
        previous_pitch_class != dominant_pitch_class
        or final_pitch_class != tonic_pitch_class
    ):
        return _failure(diagnostics, reason="wrong_root_motion")

    return SuccessResult(
        success=True,
        diagnostics={
            **diagnostics,
            "reason": "success",
        },
    )


def rank2_lifted_cadence_success(
    context: TowerRewardContext,
) -> SuccessResult:
    """Return whether a rank-2 window lifts the rank-1 cadence with thirds."""
    if context.rank != 2:
        raise ValueError("rank2_lifted_cadence_success requires rank 2 context")

    diagnostics: dict[str, object] = {
        "rank": context.rank,
        "kind": "rank2_lifted_cadence_success",
    }

    parent_context = _project_rank2_context(context)
    parent_result = rank1_projected_cadence_success(parent_context)
    diagnostics["parent"] = parent_result.diagnostics
    if not parent_result.success:
        return _failure(diagnostics, reason="parent_success_failed")

    key_pitch_class = context.key_pitch_class
    if key_pitch_class is None:
        return _failure(diagnostics, reason="missing_key_pitch_class")

    previous_state = context.source
    final_state = context.target
    dominant_third_pitch_class = (key_pitch_class + 7 + 4) % 12
    tonic_third_pitch_class = (key_pitch_class + 4) % 12
    previous_outer_pitch_class = previous_state[1] % 12
    final_outer_pitch_class = final_state[1] % 12

    diagnostics.update(
        {
            "previous_outer_pitch_class": previous_outer_pitch_class,
            "final_outer_pitch_class": final_outer_pitch_class,
            "dominant_third_pitch_class": dominant_third_pitch_class,
            "tonic_third_pitch_class": tonic_third_pitch_class,
        }
    )

    if previous_outer_pitch_class != dominant_third_pitch_class:
        return _failure(diagnostics, reason="wrong_dominant_outer_third")
    if final_outer_pitch_class != tonic_third_pitch_class:
        return _failure(diagnostics, reason="wrong_tonic_outer_third")

    return SuccessResult(
        success=True,
        diagnostics={
            **diagnostics,
            "reason": "success",
        },
    )


def rank3_triadic_cadence_success(
    context: TowerRewardContext,
) -> SuccessResult:
    """Return whether a rank-3 window realizes the accepted triadic cadence."""
    if context.rank != 3:
        raise ValueError("rank3_triadic_cadence_success requires rank 3 context")

    diagnostics: dict[str, object] = {
        "rank": context.rank,
        "kind": "rank3_triadic_cadence_success",
    }

    parent_context = _project_rank3_context(context)
    parent_result = rank2_lifted_cadence_success(parent_context)
    diagnostics["parent"] = parent_result.diagnostics
    if not parent_result.success:
        return _failure(diagnostics, reason="parent_success_failed")

    if context.target_root_octave is None:
        return _failure(diagnostics, reason="missing_target_root_octave")
    if context.key_pitch_class is None:
        return _failure(diagnostics, reason="missing_key_pitch_class")

    final_pedal_octave = midi_to_octave(context.target[0])
    diagnostics["final_pedal_octave"] = final_pedal_octave
    diagnostics["target_root_octave"] = context.target_root_octave
    if final_pedal_octave != context.target_root_octave:
        return _failure(diagnostics, reason="wrong_pedal_octave")

    dominant_inner_pitch_class = (context.key_pitch_class + 2) % 12
    tonic_inner_pitch_class = (context.key_pitch_class + 7) % 12
    previous_inner_pitch_class = context.source[1] % 12
    final_inner_pitch_class = context.target[1] % 12
    diagnostics.update(
        {
            "previous_inner_pitch_class": previous_inner_pitch_class,
            "final_inner_pitch_class": final_inner_pitch_class,
            "dominant_inner_pitch_class": dominant_inner_pitch_class,
            "tonic_inner_pitch_class": tonic_inner_pitch_class,
        }
    )

    if previous_inner_pitch_class != dominant_inner_pitch_class:
        return _failure(diagnostics, reason="wrong_dominant_inner_voice")
    if final_inner_pitch_class != tonic_inner_pitch_class:
        return _failure(diagnostics, reason="wrong_tonic_inner_voice")

    return SuccessResult(
        success=True,
        diagnostics={
            **diagnostics,
            "reason": "success",
        },
    )


def _valid_window_states(context: TowerRewardContext) -> tuple[TowerState, ...]:
    return tuple(
        state
        for state, is_valid in zip(
            context.window.states,
            context.window.valid_mask,
            strict=True,
        )
        if is_valid
    )


def _project_rank2_context(context: TowerRewardContext) -> TowerRewardContext:
    return TowerRewardContext(
        rank=1,
        step_index=context.step_index,
        source=project_state(context.source),
        target=project_state(context.target),
        action=project_action(context.action),
        window=project_window(context.window),
        measure_size=context.measure_size,
        max_steps=context.max_steps,
        max_step_size=context.max_step_size,
        key_pitch_class=context.key_pitch_class,
        target_root_octave=context.target_root_octave,
        is_final_step=context.is_final_step,
        diagnostics={
            "projected_from_rank": context.rank,
        },
    )


def _project_rank3_context(context: TowerRewardContext) -> TowerRewardContext:
    return TowerRewardContext(
        rank=2,
        step_index=context.step_index,
        source=project_state(context.source),
        target=project_state(context.target),
        action=project_action(context.action),
        window=project_window(context.window),
        measure_size=context.measure_size,
        max_steps=context.max_steps,
        max_step_size=context.max_step_size,
        key_pitch_class=context.key_pitch_class,
        target_root_octave=context.target_root_octave,
        is_final_step=context.is_final_step,
        diagnostics={
            "projected_from_rank": context.rank,
        },
    )


def _failure(
    diagnostics: dict[str, object],
    *,
    reason: str,
    **extra: object,
) -> SuccessResult:
    return SuccessResult(
        success=False,
        diagnostics={
            **diagnostics,
            **extra,
            "reason": reason,
        },
    )
