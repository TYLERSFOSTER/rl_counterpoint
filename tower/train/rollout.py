"""Rank-local rollout helpers for the tower model."""

from __future__ import annotations

from collections.abc import Callable

from tower.action.assembly import assemble_action, new_voice_index
from tower.graph.actions import active_lift_choices, legal_actions_for_state
from tower.graph.legality import is_valid_transition
from tower.graph.projection import project_state, project_window
from tower.graph.spec import TowerGraphSpec
from tower.policy.samplers import SamplerResult
from tower.reward.context import NewFacts, TowerRewardContext
from tower.reward.result import TowerRewardResult
from tower.state_action import TowerAction, TowerState, apply_action, validate_state
from tower.train.trajectory import (
    TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
    TRAJECTORY_OUTCOME_INVALID_EXTENSION,
    TRAJECTORY_OUTCOME_PARENT_FAILURE,
    TRAJECTORY_OUTCOME_TAIL_STAGNATION,
    TRAJECTORY_OUTCOME_VALID,
    TowerTrajectory,
    TowerTrajectoryStep,
)
from tower.window import build_window

ParentSampler = Callable[..., SamplerResult[TowerAction]]
ActiveSampler = Callable[..., SamplerResult[int]]
RewardFunction = Callable[[TowerRewardContext], TowerRewardResult]


def rollout_rank1(
    *,
    initial_state: TowerState,
    max_steps: int,
    active_sampler: ActiveSampler,
    reward_fn: RewardFunction,
    graph_spec: TowerGraphSpec | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    goal_octave_cadence_window_measures: int | None = 2,
    tail_stagnation_patience: int = 4,
    tail_stagnation_reward: TowerRewardResult | None = None,
) -> TowerTrajectory:
    """Roll out a rank-1 trajectory under an active sampler."""
    spec = TowerGraphSpec(rank=1) if graph_spec is None else graph_spec
    if spec.rank != 1:
        raise ValueError("rollout_rank1 requires a rank-1 graph spec")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if (
        goal_octave_cadence_window_measures is not None
        and goal_octave_cadence_window_measures < 1
    ):
        raise ValueError("goal_octave_cadence_window_measures must be at least 1")
    if tail_stagnation_patience < 2:
        raise ValueError("tail_stagnation_patience must be at least 2")
    validate_state(initial_state, rank=1)

    history = [initial_state]
    steps: list[TowerTrajectoryStep] = []
    goal_octave_entry_step_index: int | None = None

    for step_index in range(max_steps):
        source_state = history[-1]
        window = build_window(
            history=tuple(history),
            step_index=step_index,
            measure_size=measure_size,
            context_measures=context_measures,
        )
        choices = tuple(action[0] for action in legal_actions_for_state(state=source_state, spec=spec))
        active_result = active_sampler(
            rank=1,
            step_index=step_index,
            state=source_state,
            window=window,
            active_choices=choices,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=spec.max_step_size,
        )
        active_choice = active_result.choice
        if active_choice not in choices:
            raise ValueError("active choice must be legal for rank 1")

        assembled_action = assemble_action(
            rank=1,
            parent_action=None,
            new_action=active_choice,
        )
        attempted_target_state = apply_action(source_state, assembled_action)
        realized_next_state = attempted_target_state
        entered_goal_octave = (
            not _state_is_in_target_octave(source_state, target_root_octave)
            and _state_is_in_target_octave(realized_next_state, target_root_octave)
        )
        if goal_octave_entry_step_index is None and entered_goal_octave:
            goal_octave_entry_step_index = step_index
        goal_deadline_step_index = _goal_octave_deadline_step_index(
            goal_octave_entry_step_index=goal_octave_entry_step_index,
            measure_size=measure_size,
            goal_octave_cadence_window_measures=goal_octave_cadence_window_measures,
        )
        tail_progress_score = _rank1_terminal_progress_score(
            state=realized_next_state,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
        )
        tail_stagnation_detected, tail_stagnation_kind = _detect_tail_stagnation(
            state_history=tuple(history) + (realized_next_state,),
            progress_score_fn=lambda state: _rank1_terminal_progress_score(
                state=state,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
            ),
            goal_octave_entry_step_index=goal_octave_entry_step_index,
            current_step_index=step_index,
            tail_stagnation_patience=tail_stagnation_patience,
            detect_two_cycle=False,
            detect_three_cycle=False,
        )
        cadence_check_step = (
            goal_octave_entry_step_index is not None
            and step_index % measure_size == measure_size - 1
            and (
                goal_deadline_step_index is None
                or step_index <= goal_deadline_step_index
            )
        )
        goal_window_deadline_step = (
            goal_deadline_step_index is not None
            and step_index >= goal_deadline_step_index
        )
        is_final_step = (
            step_index == max_steps - 1
            or cadence_check_step
            or goal_window_deadline_step
            or tail_stagnation_detected
        )
        reward = reward_fn(
            TowerRewardContext(
                rank=1,
                step_index=step_index,
                source=source_state,
                target=realized_next_state,
                action=assembled_action,
                window=window,
                measure_size=measure_size,
                max_steps=max_steps,
                max_step_size=spec.max_step_size,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
                is_final_step=is_final_step,
                new_facts=NewFacts(
                    new_voice_index=new_voice_index(rank=1),
                    new_action=active_choice,
                ),
                diagnostics={
                    "goal_octave_entry_step_index": goal_octave_entry_step_index,
                    "goal_octave_deadline_step_index": goal_deadline_step_index,
                    "goal_octave_cadence_window_measures": (
                        goal_octave_cadence_window_measures
                    ),
                    "entered_goal_octave": entered_goal_octave,
                    "cadence_check_step": cadence_check_step,
                    "goal_octave_deadline_step": goal_window_deadline_step,
                    "terminal_regime_active": goal_octave_entry_step_index is not None,
                    "tail_progress_score": tail_progress_score,
                    "tail_stagnation_detected": tail_stagnation_detected,
                    "tail_stagnation_kind": tail_stagnation_kind,
                },
            )
        )
        if not isinstance(reward, TowerRewardResult):
            raise TypeError("reward_fn must return a TowerRewardResult")

        outcome = TRAJECTORY_OUTCOME_VALID
        if tail_stagnation_detected and not reward.is_terminal_success:
            reward = _default_outcome_reward(
                provided=tail_stagnation_reward,
                outcome=TRAJECTORY_OUTCOME_TAIL_STAGNATION,
            )
            outcome = TRAJECTORY_OUTCOME_TAIL_STAGNATION

        terminated = reward.is_terminal_success
        goal_window_expired = goal_window_deadline_step
        truncated = not terminated and (
            step_index == max_steps - 1
            or goal_window_expired
            or tail_stagnation_detected
        )
        step = TowerTrajectoryStep(
            rank=1,
            step_index=step_index,
            source_state=source_state,
            window=window,
            parent_state=None,
            parent_action=None,
            active_choice=active_choice,
            assembled_action=assembled_action,
            attempted_target_state=attempted_target_state,
            realized_next_state=realized_next_state,
            active_logprob=active_result.logprob,
            parent_logprob=None,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            outcome=outcome,
            diagnostics={
                "active_choices": choices,
                "active_sampler": active_result.diagnostics,
                "goal_octave_entry_step_index": goal_octave_entry_step_index,
                "goal_octave_deadline_step_index": goal_deadline_step_index,
                "goal_octave_window_expired": goal_window_expired,
                "entered_goal_octave": entered_goal_octave,
                "cadence_check_step": cadence_check_step,
                "goal_octave_deadline_step": goal_window_deadline_step,
                "terminal_regime_active": goal_octave_entry_step_index is not None,
                "tail_progress_score": tail_progress_score,
                "tail_stagnation_detected": tail_stagnation_detected,
                "tail_stagnation_kind": tail_stagnation_kind,
            },
        )
        steps.append(step)
        history.append(realized_next_state)

        if terminated or truncated:
            break

    return TowerTrajectory(steps=tuple(steps))


def _state_is_in_target_octave(
    state: TowerState,
    target_root_octave: int | None,
) -> bool:
    if target_root_octave is None:
        return False
    return _midi_to_octave(state[0]) == target_root_octave


def _midi_to_octave(pitch: int) -> int:
    return pitch // 12 - 1


def _goal_octave_deadline_step_index(
    *,
    goal_octave_entry_step_index: int | None,
    measure_size: int,
    goal_octave_cadence_window_measures: int | None,
) -> int | None:
    if (
        goal_octave_entry_step_index is None
        or goal_octave_cadence_window_measures is None
    ):
        return None
    return (
        goal_octave_entry_step_index
        + measure_size * goal_octave_cadence_window_measures
        - 1
    )


def rollout_rank2(
    *,
    initial_state: TowerState,
    max_steps: int,
    parent_sampler: ParentSampler,
    active_sampler: ActiveSampler,
    reward_fn: RewardFunction,
    graph_spec: TowerGraphSpec | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    goal_octave_cadence_window_measures: int | None = 2,
    tail_stagnation_patience: int = 4,
    invalid_extension_reward: TowerRewardResult | None = None,
    empty_lift_fiber_reward: TowerRewardResult | None = None,
    parent_failure_reward: TowerRewardResult | None = None,
    tail_stagnation_reward: TowerRewardResult | None = None,
) -> TowerTrajectory:
    """Roll out a rank-2 trajectory under scripted happy-path samplers."""
    spec = TowerGraphSpec(rank=2) if graph_spec is None else graph_spec
    if spec.rank != 2:
        raise ValueError("rollout_rank2 requires a rank-2 graph spec")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if (
        goal_octave_cadence_window_measures is not None
        and goal_octave_cadence_window_measures < 1
    ):
        raise ValueError("goal_octave_cadence_window_measures must be at least 1")
    if tail_stagnation_patience < 2:
        raise ValueError("tail_stagnation_patience must be at least 2")
    validate_state(initial_state, rank=2)

    history = [initial_state]
    steps: list[TowerTrajectoryStep] = []
    goal_octave_entry_step_index: int | None = None

    for step_index in range(max_steps):
        source_state = history[-1]
        window = build_window(
            history=tuple(history),
            step_index=step_index,
            measure_size=measure_size,
            context_measures=context_measures,
        )
        parent_state = project_state(source_state)
        parent_window = project_window(window)
        parent_spec = TowerGraphSpec(
            rank=1,
            key_pitch_class=spec.key_pitch_class,
            pitch_min=spec.pitch_min,
            pitch_max=spec.pitch_max,
            max_step_size=spec.max_step_size,
        )

        parent_result = parent_sampler(
            rank=1,
            step_index=step_index,
            state=parent_state,
            full_state=source_state,
            window=parent_window,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=parent_spec.max_step_size,
        )
        parent_action = parent_result.choice
        if not is_valid_transition(parent_state, parent_action, parent_spec):
            reward = _default_outcome_reward(
                provided=parent_failure_reward,
                outcome=TRAJECTORY_OUTCOME_PARENT_FAILURE,
            )
            step = TowerTrajectoryStep(
                rank=2,
                step_index=step_index,
                source_state=source_state,
                window=window,
                parent_state=parent_state,
                parent_action=parent_action,
                active_choice=None,
                assembled_action=(0, 0),
                attempted_target_state=source_state,
                realized_next_state=source_state,
                active_logprob=None,
                parent_logprob=parent_result.logprob,
                reward=reward,
                terminated=False,
                truncated=True,
                outcome=TRAJECTORY_OUTCOME_PARENT_FAILURE,
                diagnostics={
                    "parent_sampler": parent_result.diagnostics,
                    "parent_failure": True,
                },
            )
            steps.append(step)
            history.append(source_state)
            break

        choices = active_lift_choices(
            state=source_state,
            parent_action=parent_action,
            spec=spec,
        )
        if not choices:
            reward = _default_outcome_reward(
                provided=empty_lift_fiber_reward,
                outcome=TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
            )
            step = TowerTrajectoryStep(
                rank=2,
                step_index=step_index,
                source_state=source_state,
                window=window,
                parent_state=parent_state,
                parent_action=parent_action,
                active_choice=None,
                assembled_action=(0, 0),
                attempted_target_state=source_state,
                realized_next_state=source_state,
                active_logprob=None,
                parent_logprob=parent_result.logprob,
                reward=reward,
                terminated=False,
                truncated=step_index == max_steps - 1,
                outcome=TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
                diagnostics={
                    "active_choices": choices,
                    "parent_sampler": parent_result.diagnostics,
                    "empty_lift_fiber": True,
                },
            )
            steps.append(step)
            history.append(source_state)
            if step.truncated:
                break
            continue

        active_result = active_sampler(
            rank=2,
            step_index=step_index,
            state=source_state,
            window=window,
            parent_state=parent_state,
            parent_action=parent_action,
            active_choices=choices,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=spec.max_step_size,
        )
        active_choice = active_result.choice
        if active_choice not in choices:
            assembled_action = assemble_action(
                rank=2,
                parent_action=parent_action,
                new_action=active_choice,
            )
            attempted_target_state = apply_action(source_state, assembled_action)
            reward = _default_outcome_reward(
                provided=invalid_extension_reward,
                outcome=TRAJECTORY_OUTCOME_INVALID_EXTENSION,
            )
            step = TowerTrajectoryStep(
                rank=2,
                step_index=step_index,
                source_state=source_state,
                window=window,
                parent_state=parent_state,
                parent_action=parent_action,
                active_choice=active_choice,
                assembled_action=assembled_action,
                attempted_target_state=attempted_target_state,
                realized_next_state=source_state,
                active_logprob=active_result.logprob,
                parent_logprob=parent_result.logprob,
                reward=reward,
                terminated=False,
                truncated=step_index == max_steps - 1,
                outcome=TRAJECTORY_OUTCOME_INVALID_EXTENSION,
                diagnostics={
                    "active_choices": choices,
                    "parent_sampler": parent_result.diagnostics,
                    "active_sampler": active_result.diagnostics,
                    "invalid_extension": True,
                },
            )
            steps.append(step)
            history.append(source_state)
            if step.truncated:
                break
            continue

        assembled_action = assemble_action(
            rank=2,
            parent_action=parent_action,
            new_action=active_choice,
        )
        if not is_valid_transition(source_state, assembled_action, spec):
            raise ValueError("assembled action must define a valid transition")

        attempted_target_state = apply_action(source_state, assembled_action)
        realized_next_state = attempted_target_state
        entered_goal_octave = (
            not _state_is_in_target_octave(source_state, target_root_octave)
            and _state_is_in_target_octave(realized_next_state, target_root_octave)
        )
        if goal_octave_entry_step_index is None and entered_goal_octave:
            goal_octave_entry_step_index = step_index
        goal_deadline_step_index = _goal_octave_deadline_step_index(
            goal_octave_entry_step_index=goal_octave_entry_step_index,
            measure_size=measure_size,
            goal_octave_cadence_window_measures=goal_octave_cadence_window_measures,
        )
        tail_progress_score = _rank2_terminal_progress_score(
            state=realized_next_state,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
        )
        tail_stagnation_detected, tail_stagnation_kind = _detect_tail_stagnation(
            state_history=tuple(history) + (realized_next_state,),
            progress_score_fn=lambda state: _rank2_terminal_progress_score(
                state=state,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
            ),
            goal_octave_entry_step_index=goal_octave_entry_step_index,
            current_step_index=step_index,
            tail_stagnation_patience=tail_stagnation_patience,
            detect_two_cycle=True,
            detect_three_cycle=True,
        )
        cadence_check_step = (
            goal_octave_entry_step_index is not None
            and step_index % measure_size == measure_size - 1
            and (
                goal_deadline_step_index is None
                or step_index <= goal_deadline_step_index
            )
        )
        goal_window_deadline_step = (
            goal_deadline_step_index is not None
            and step_index >= goal_deadline_step_index
        )
        reward = reward_fn(
            TowerRewardContext(
                rank=2,
                step_index=step_index,
                source=source_state,
                target=realized_next_state,
                action=assembled_action,
                window=window,
                measure_size=measure_size,
                max_steps=max_steps,
                max_step_size=spec.max_step_size,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
                is_final_step=(
                    step_index == max_steps - 1
                    or cadence_check_step
                    or goal_window_deadline_step
                    or tail_stagnation_detected
                ),
                new_facts=NewFacts(
                    new_voice_index=new_voice_index(rank=2),
                    new_action=active_choice,
                ),
                diagnostics={
                    "goal_octave_entry_step_index": goal_octave_entry_step_index,
                    "goal_octave_deadline_step_index": goal_deadline_step_index,
                    "goal_octave_cadence_window_measures": (
                        goal_octave_cadence_window_measures
                    ),
                    "entered_goal_octave": entered_goal_octave,
                    "cadence_check_step": cadence_check_step,
                    "goal_octave_deadline_step": goal_window_deadline_step,
                    "terminal_regime_active": goal_octave_entry_step_index is not None,
                    "tail_progress_score": tail_progress_score,
                    "tail_stagnation_detected": tail_stagnation_detected,
                    "tail_stagnation_kind": tail_stagnation_kind,
                },
            )
        )
        if not isinstance(reward, TowerRewardResult):
            raise TypeError("reward_fn must return a TowerRewardResult")

        outcome = TRAJECTORY_OUTCOME_VALID
        if tail_stagnation_detected and not reward.is_terminal_success:
            reward = _default_outcome_reward(
                provided=tail_stagnation_reward,
                outcome=TRAJECTORY_OUTCOME_TAIL_STAGNATION,
            )
            outcome = TRAJECTORY_OUTCOME_TAIL_STAGNATION

        terminated = reward.is_terminal_success
        truncated = not terminated and (
            step_index == max_steps - 1
            or goal_window_deadline_step
            or tail_stagnation_detected
        )
        step = TowerTrajectoryStep(
            rank=2,
            step_index=step_index,
            source_state=source_state,
            window=window,
            parent_state=parent_state,
            parent_action=parent_action,
            active_choice=active_choice,
            assembled_action=assembled_action,
            attempted_target_state=attempted_target_state,
            realized_next_state=realized_next_state,
            active_logprob=active_result.logprob,
            parent_logprob=parent_result.logprob,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            outcome=outcome,
            diagnostics={
                "active_choices": choices,
                "parent_sampler": parent_result.diagnostics,
                "active_sampler": active_result.diagnostics,
                "goal_octave_entry_step_index": goal_octave_entry_step_index,
                "goal_octave_deadline_step_index": goal_deadline_step_index,
                "goal_octave_window_expired": goal_window_deadline_step,
                "entered_goal_octave": entered_goal_octave,
                "cadence_check_step": cadence_check_step,
                "goal_octave_deadline_step": goal_window_deadline_step,
                "terminal_regime_active": goal_octave_entry_step_index is not None,
                "tail_progress_score": tail_progress_score,
                "tail_stagnation_detected": tail_stagnation_detected,
                "tail_stagnation_kind": tail_stagnation_kind,
            },
        )
        steps.append(step)
        history.append(realized_next_state)

        if terminated or truncated:
            break

    return TowerTrajectory(steps=tuple(steps))


def rollout_rank3(
    *,
    initial_state: TowerState,
    max_steps: int,
    parent_sampler: ParentSampler,
    active_sampler: ActiveSampler,
    reward_fn: RewardFunction,
    graph_spec: TowerGraphSpec | None = None,
    measure_size: int = 4,
    context_measures: int = 2,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    goal_octave_cadence_window_measures: int | None = 2,
    tail_stagnation_patience: int = 4,
    invalid_extension_reward: TowerRewardResult | None = None,
    empty_lift_fiber_reward: TowerRewardResult | None = None,
    parent_failure_reward: TowerRewardResult | None = None,
    tail_stagnation_reward: TowerRewardResult | None = None,
) -> TowerTrajectory:
    """Roll out a rank-3 trajectory over a scripted rank-2 parent action stream."""
    spec = TowerGraphSpec(rank=3) if graph_spec is None else graph_spec
    if spec.rank != 3:
        raise ValueError("rollout_rank3 requires a rank-3 graph spec")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if (
        goal_octave_cadence_window_measures is not None
        and goal_octave_cadence_window_measures < 1
    ):
        raise ValueError("goal_octave_cadence_window_measures must be at least 1")
    if tail_stagnation_patience < 2:
        raise ValueError("tail_stagnation_patience must be at least 2")
    validate_state(initial_state, rank=3)

    history = [initial_state]
    steps: list[TowerTrajectoryStep] = []
    goal_octave_entry_step_index: int | None = None

    for step_index in range(max_steps):
        source_state = history[-1]
        window = build_window(
            history=tuple(history),
            step_index=step_index,
            measure_size=measure_size,
            context_measures=context_measures,
        )
        parent_state = project_state(source_state)
        parent_window = project_window(window)
        parent_spec = TowerGraphSpec(
            rank=2,
            key_pitch_class=spec.key_pitch_class,
            pitch_min=spec.pitch_min,
            pitch_max=spec.pitch_max,
            max_step_size=spec.max_step_size,
        )

        parent_result = parent_sampler(
            rank=2,
            step_index=step_index,
            state=parent_state,
            full_state=source_state,
            window=parent_window,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=parent_spec.max_step_size,
        )
        parent_action = parent_result.choice
        if not is_valid_transition(parent_state, parent_action, parent_spec):
            reward = _default_outcome_reward(
                provided=parent_failure_reward,
                outcome=TRAJECTORY_OUTCOME_PARENT_FAILURE,
            )
            step = TowerTrajectoryStep(
                rank=3,
                step_index=step_index,
                source_state=source_state,
                window=window,
                parent_state=parent_state,
                parent_action=parent_action,
                active_choice=None,
                assembled_action=(0, 0, 0),
                attempted_target_state=source_state,
                realized_next_state=source_state,
                active_logprob=None,
                parent_logprob=parent_result.logprob,
                reward=reward,
                terminated=False,
                truncated=True,
                outcome=TRAJECTORY_OUTCOME_PARENT_FAILURE,
                diagnostics={
                    "parent_sampler": parent_result.diagnostics,
                    "parent_failure": True,
                },
            )
            steps.append(step)
            history.append(source_state)
            break

        choices = active_lift_choices(
            state=source_state,
            parent_action=parent_action,
            spec=spec,
        )
        if not choices:
            reward = _default_outcome_reward(
                provided=empty_lift_fiber_reward,
                outcome=TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
            )
            step = TowerTrajectoryStep(
                rank=3,
                step_index=step_index,
                source_state=source_state,
                window=window,
                parent_state=parent_state,
                parent_action=parent_action,
                active_choice=None,
                assembled_action=(0, 0, 0),
                attempted_target_state=source_state,
                realized_next_state=source_state,
                active_logprob=None,
                parent_logprob=parent_result.logprob,
                reward=reward,
                terminated=False,
                truncated=step_index == max_steps - 1,
                outcome=TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
                diagnostics={
                    "active_choices": choices,
                    "parent_sampler": parent_result.diagnostics,
                    "empty_lift_fiber": True,
                },
            )
            steps.append(step)
            history.append(source_state)
            if step.truncated:
                break
            continue

        active_result = active_sampler(
            rank=3,
            step_index=step_index,
            state=source_state,
            window=window,
            parent_state=parent_state,
            parent_action=parent_action,
            active_choices=choices,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
            max_step_size=spec.max_step_size,
        )
        active_choice = active_result.choice
        if active_choice not in choices:
            assembled_action = assemble_action(
                rank=3,
                parent_action=parent_action,
                new_action=active_choice,
            )
            attempted_target_state = apply_action(source_state, assembled_action)
            reward = _default_outcome_reward(
                provided=invalid_extension_reward,
                outcome=TRAJECTORY_OUTCOME_INVALID_EXTENSION,
            )
            step = TowerTrajectoryStep(
                rank=3,
                step_index=step_index,
                source_state=source_state,
                window=window,
                parent_state=parent_state,
                parent_action=parent_action,
                active_choice=active_choice,
                assembled_action=assembled_action,
                attempted_target_state=attempted_target_state,
                realized_next_state=source_state,
                active_logprob=active_result.logprob,
                parent_logprob=parent_result.logprob,
                reward=reward,
                terminated=False,
                truncated=step_index == max_steps - 1,
                outcome=TRAJECTORY_OUTCOME_INVALID_EXTENSION,
                diagnostics={
                    "active_choices": choices,
                    "parent_sampler": parent_result.diagnostics,
                    "active_sampler": active_result.diagnostics,
                    "invalid_extension": True,
                },
            )
            steps.append(step)
            history.append(source_state)
            if step.truncated:
                break
            continue

        assembled_action = assemble_action(
            rank=3,
            parent_action=parent_action,
            new_action=active_choice,
        )
        if not is_valid_transition(source_state, assembled_action, spec):
            raise ValueError("assembled action must define a valid transition")

        attempted_target_state = apply_action(source_state, assembled_action)
        realized_next_state = attempted_target_state
        entered_goal_octave = (
            not _state_is_in_target_octave(source_state, target_root_octave)
            and _state_is_in_target_octave(realized_next_state, target_root_octave)
        )
        if goal_octave_entry_step_index is None and entered_goal_octave:
            goal_octave_entry_step_index = step_index
        goal_deadline_step_index = _goal_octave_deadline_step_index(
            goal_octave_entry_step_index=goal_octave_entry_step_index,
            measure_size=measure_size,
            goal_octave_cadence_window_measures=goal_octave_cadence_window_measures,
        )
        tail_progress_score = _rank3_terminal_progress_score(
            state=realized_next_state,
            key_pitch_class=key_pitch_class,
            target_root_octave=target_root_octave,
        )
        tail_stagnation_detected, tail_stagnation_kind = _detect_tail_stagnation(
            state_history=tuple(history) + (realized_next_state,),
            progress_score_fn=lambda state: _rank3_terminal_progress_score(
                state=state,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
            ),
            goal_octave_entry_step_index=goal_octave_entry_step_index,
            current_step_index=step_index,
            tail_stagnation_patience=tail_stagnation_patience,
            detect_two_cycle=True,
            detect_three_cycle=True,
        )
        cadence_check_step = (
            goal_octave_entry_step_index is not None
            and step_index % measure_size == measure_size - 1
            and (
                goal_deadline_step_index is None
                or step_index <= goal_deadline_step_index
            )
        )
        goal_window_deadline_step = (
            goal_deadline_step_index is not None
            and step_index >= goal_deadline_step_index
        )
        reward = reward_fn(
            TowerRewardContext(
                rank=3,
                step_index=step_index,
                source=source_state,
                target=realized_next_state,
                action=assembled_action,
                window=window,
                measure_size=measure_size,
                max_steps=max_steps,
                max_step_size=spec.max_step_size,
                key_pitch_class=key_pitch_class,
                target_root_octave=target_root_octave,
                is_final_step=(
                    step_index == max_steps - 1
                    or cadence_check_step
                    or goal_window_deadline_step
                    or tail_stagnation_detected
                ),
                new_facts=NewFacts(
                    new_voice_index=new_voice_index(rank=3),
                    new_action=active_choice,
                    full_sonority_used=True,
                ),
                diagnostics={
                    "goal_octave_entry_step_index": goal_octave_entry_step_index,
                    "goal_octave_deadline_step_index": goal_deadline_step_index,
                    "goal_octave_cadence_window_measures": (
                        goal_octave_cadence_window_measures
                    ),
                    "entered_goal_octave": entered_goal_octave,
                    "cadence_check_step": cadence_check_step,
                    "goal_octave_deadline_step": goal_window_deadline_step,
                    "terminal_regime_active": goal_octave_entry_step_index is not None,
                    "tail_progress_score": tail_progress_score,
                    "tail_stagnation_detected": tail_stagnation_detected,
                    "tail_stagnation_kind": tail_stagnation_kind,
                },
            )
        )
        if not isinstance(reward, TowerRewardResult):
            raise TypeError("reward_fn must return a TowerRewardResult")

        outcome = TRAJECTORY_OUTCOME_VALID
        if tail_stagnation_detected and not reward.is_terminal_success:
            reward = _default_outcome_reward(
                provided=tail_stagnation_reward,
                outcome=TRAJECTORY_OUTCOME_TAIL_STAGNATION,
            )
            outcome = TRAJECTORY_OUTCOME_TAIL_STAGNATION

        terminated = reward.is_terminal_success
        truncated = not terminated and (
            step_index == max_steps - 1
            or goal_window_deadline_step
            or tail_stagnation_detected
        )
        step = TowerTrajectoryStep(
            rank=3,
            step_index=step_index,
            source_state=source_state,
            window=window,
            parent_state=parent_state,
            parent_action=parent_action,
            active_choice=active_choice,
            assembled_action=assembled_action,
            attempted_target_state=attempted_target_state,
            realized_next_state=realized_next_state,
            active_logprob=active_result.logprob,
            parent_logprob=parent_result.logprob,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            outcome=outcome,
            diagnostics={
                "active_choices": choices,
                "parent_sampler": parent_result.diagnostics,
                "active_sampler": active_result.diagnostics,
                "goal_octave_entry_step_index": goal_octave_entry_step_index,
                "goal_octave_deadline_step_index": goal_deadline_step_index,
                "goal_octave_window_expired": goal_window_deadline_step,
                "entered_goal_octave": entered_goal_octave,
                "cadence_check_step": cadence_check_step,
                "goal_octave_deadline_step": goal_window_deadline_step,
                "terminal_regime_active": goal_octave_entry_step_index is not None,
                "tail_progress_score": tail_progress_score,
                "tail_stagnation_detected": tail_stagnation_detected,
                "tail_stagnation_kind": tail_stagnation_kind,
            },
        )
        steps.append(step)
        history.append(realized_next_state)

        if terminated or truncated:
            break

    return TowerTrajectory(steps=tuple(steps))


def _rank1_terminal_progress_score(
    *,
    state: TowerState,
    key_pitch_class: int | None,
    target_root_octave: int | None,
) -> int:
    if key_pitch_class is None or target_root_octave is None:
        return 0
    score = 0
    if state[0] % 12 == key_pitch_class % 12:
        score += 1
    if _state_is_in_target_octave(state, target_root_octave):
        score += 1
    return score


def _rank2_terminal_progress_score(
    *,
    state: TowerState,
    key_pitch_class: int | None,
    target_root_octave: int | None,
) -> int:
    if key_pitch_class is None or target_root_octave is None:
        return 0
    tonic_pitch_class = key_pitch_class % 12
    tonic_third_pitch_class = (key_pitch_class + 4) % 12
    score = 0
    if state[0] % 12 == tonic_pitch_class:
        score += 1
    if state[1] % 12 == tonic_third_pitch_class:
        score += 1
    if _state_is_in_target_octave(state, target_root_octave):
        score += 1
    return score


def _rank3_terminal_progress_score(
    *,
    state: TowerState,
    key_pitch_class: int | None,
    target_root_octave: int | None,
) -> int:
    if key_pitch_class is None or target_root_octave is None:
        return 0
    tonic_pitch_class = key_pitch_class % 12
    tonic_inner_pitch_class = (key_pitch_class + 7) % 12
    tonic_outer_pitch_class = (key_pitch_class + 4) % 12
    score = 0
    if state[0] % 12 == tonic_pitch_class:
        score += 1
    if state[1] % 12 == tonic_inner_pitch_class:
        score += 1
    if state[2] % 12 == tonic_outer_pitch_class:
        score += 1
    if _state_is_in_target_octave(state, target_root_octave):
        score += 1
    return score


def _detect_tail_stagnation(
    *,
    state_history: tuple[TowerState, ...],
    progress_score_fn: Callable[[TowerState], int],
    goal_octave_entry_step_index: int | None,
    current_step_index: int,
    tail_stagnation_patience: int,
    detect_two_cycle: bool,
    detect_three_cycle: bool,
) -> tuple[bool, str | None]:
    if goal_octave_entry_step_index is None:
        return False, None
    steps_since_entry = current_step_index - goal_octave_entry_step_index + 1
    if steps_since_entry < tail_stagnation_patience:
        return False, None

    terminal_states = state_history[goal_octave_entry_step_index + 1 :]
    if len(terminal_states) < tail_stagnation_patience:
        return False, None

    if detect_two_cycle and len(terminal_states) >= 3:
        cycle2_states = terminal_states[-3:]
        cycle2_scores = tuple(progress_score_fn(state) for state in cycle2_states)
        if (
            cycle2_states[0] == cycle2_states[2]
            and cycle2_states[0] != cycle2_states[1]
            and max(cycle2_scores) <= cycle2_scores[0]
        ):
            return True, "two_cycle"

    if detect_three_cycle and len(terminal_states) >= 6:
        cycle3_states = terminal_states[-6:]
        cycle3_scores = tuple(progress_score_fn(state) for state in cycle3_states)
        if (
            cycle3_states[0] == cycle3_states[3]
            and cycle3_states[1] == cycle3_states[4]
            and cycle3_states[2] == cycle3_states[5]
            and len({cycle3_states[0], cycle3_states[1], cycle3_states[2]}) == 3
            and max(cycle3_scores) <= cycle3_scores[0]
        ):
            return True, "three_cycle"

    recent_states = terminal_states[-tail_stagnation_patience:]
    recent_scores = tuple(progress_score_fn(state) for state in recent_states)
    repeated_full_state = recent_states[-1] in recent_states[:-1]
    no_progress = max(recent_scores) <= recent_scores[0]
    if repeated_full_state and no_progress:
        return True, "repeat_no_progress"

    return False, None


def _default_outcome_reward(
    *,
    provided: TowerRewardResult | None,
    outcome: str,
) -> TowerRewardResult:
    if provided is not None:
        return provided
    if outcome == TRAJECTORY_OUTCOME_TAIL_STAGNATION:
        return TowerRewardResult(
            reward=-1.0,
            hard_violation=False,
            is_terminal_success=False,
            diagnostics={"outcome": outcome},
        )
    return TowerRewardResult(
        reward=0.0,
        hard_violation=False,
        is_terminal_success=False,
        diagnostics={"outcome": outcome},
    )
