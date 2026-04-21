"""Rank-local rollout helpers for the tower model."""

from __future__ import annotations

from collections.abc import Callable

from tower.action.assembly import assemble_action, new_voice_index
from tower.graph.actions import action_space, active_lift_choices
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
) -> TowerTrajectory:
    """Roll out a rank-1 trajectory under an active sampler."""
    spec = TowerGraphSpec(rank=1) if graph_spec is None else graph_spec
    if spec.rank != 1:
        raise ValueError("rollout_rank1 requires a rank-1 graph spec")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    validate_state(initial_state, rank=1)

    history = [initial_state]
    steps: list[TowerTrajectoryStep] = []

    for step_index in range(max_steps):
        source_state = history[-1]
        window = build_window(
            history=tuple(history),
            step_index=step_index,
            measure_size=measure_size,
            context_measures=context_measures,
        )
        choices = tuple(
            action[0]
            for action in action_space(rank=1, max_step_size=spec.max_step_size)
            if is_valid_transition(source_state, action, spec)
        )
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
                is_final_step=step_index == max_steps - 1,
                new_facts=NewFacts(
                    new_voice_index=new_voice_index(rank=1),
                    new_action=active_choice,
                ),
            )
        )
        if not isinstance(reward, TowerRewardResult):
            raise TypeError("reward_fn must return a TowerRewardResult")

        terminated = reward.is_terminal_success
        truncated = not terminated and step_index == max_steps - 1
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
            outcome=TRAJECTORY_OUTCOME_VALID,
            diagnostics={
                "active_choices": choices,
                "active_sampler": active_result.diagnostics,
            },
        )
        steps.append(step)
        history.append(realized_next_state)

        if terminated or truncated:
            break

    return TowerTrajectory(steps=tuple(steps))


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
    invalid_extension_reward: TowerRewardResult | None = None,
    empty_lift_fiber_reward: TowerRewardResult | None = None,
    parent_failure_reward: TowerRewardResult | None = None,
) -> TowerTrajectory:
    """Roll out a rank-2 trajectory under scripted happy-path samplers."""
    spec = TowerGraphSpec(rank=2) if graph_spec is None else graph_spec
    if spec.rank != 2:
        raise ValueError("rollout_rank2 requires a rank-2 graph spec")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    validate_state(initial_state, rank=2)

    history = [initial_state]
    steps: list[TowerTrajectoryStep] = []

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
            pitch_min=spec.pitch_min,
            pitch_max=spec.pitch_max,
            max_step_size=spec.max_step_size,
        )

        parent_result = parent_sampler(
            rank=1,
            step_index=step_index,
            state=parent_state,
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
                is_final_step=step_index == max_steps - 1,
                new_facts=NewFacts(
                    new_voice_index=new_voice_index(rank=2),
                    new_action=active_choice,
                ),
            )
        )
        if not isinstance(reward, TowerRewardResult):
            raise TypeError("reward_fn must return a TowerRewardResult")

        terminated = reward.is_terminal_success
        truncated = not terminated and step_index == max_steps - 1
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
            outcome=TRAJECTORY_OUTCOME_VALID,
            diagnostics={
                "active_choices": choices,
                "parent_sampler": parent_result.diagnostics,
                "active_sampler": active_result.diagnostics,
            },
        )
        steps.append(step)
        history.append(realized_next_state)

        if terminated or truncated:
            break

    return TowerTrajectory(steps=tuple(steps))


def _default_outcome_reward(
    *,
    provided: TowerRewardResult | None,
    outcome: str,
) -> TowerRewardResult:
    if provided is not None:
        return provided
    return TowerRewardResult(
        reward=0.0,
        hard_violation=False,
        is_terminal_success=False,
        diagnostics={"outcome": outcome},
    )
