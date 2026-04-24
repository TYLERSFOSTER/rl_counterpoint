"""Tests for tower rank-2 happy-path rollout."""

from __future__ import annotations

import pytest
import torch

from tower.graph.projection import project_action, project_window
from tower.graph.spec import TowerGraphSpec
from tower.policy.samplers import SamplerResult, ScriptedSampler, scripted_result
from tower.reward.context import TowerRewardContext
from tower.reward.result import TowerRewardResult
from tower.train.losses import policy_gradient_loss
from tower.train.rollout import rollout_rank1, rollout_rank2
from tower.train.trajectory import (
    TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
    TRAJECTORY_OUTCOME_INVALID_EXTENSION,
    TRAJECTORY_OUTCOME_PARENT_FAILURE,
    TRAJECTORY_OUTCOME_VALID,
)


def test_rollout_rank1_one_step_records_active_tensor_logprob() -> None:
    active_logprob = torch.tensor(-0.25, requires_grad=True)
    contexts = []

    def reward_fn(context: TowerRewardContext) -> TowerRewardResult:
        contexts.append(context)
        return TowerRewardResult(reward=1.0)

    trajectory = rollout_rank1(
        initial_state=(60,),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=2),
        active_sampler=ScriptedSampler(script=(scripted_result(1, logprob=active_logprob),)),
        reward_fn=reward_fn,
        measure_size=4,
        context_measures=1,
    )

    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.rank == 1
    assert step.parent_state is None
    assert step.parent_action is None
    assert step.active_choice == 1
    assert step.assembled_action == (1,)
    assert step.attempted_target_state == (61,)
    assert step.realized_next_state == (61,)
    assert step.active_logprob is active_logprob
    assert step.parent_logprob is None
    assert step.truncated
    assert contexts[0].new_facts.new_voice_index == 0
    assert contexts[0].new_facts.new_action == 1


def test_rollout_rank1_multi_step_advances_history_and_final_state() -> None:
    trajectory = rollout_rank1(
        initial_state=(60,),
        max_steps=2,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=2),
        active_sampler=ScriptedSampler(script=(1, -1)),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        measure_size=4,
        context_measures=1,
    )

    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].source_state == (60,)
    assert trajectory.steps[0].realized_next_state == (61,)
    assert trajectory.steps[1].source_state == (61,)
    assert trajectory.steps[1].realized_next_state == (60,)
    assert trajectory.steps[1].window.states[-2:] == ((60,), (61,))
    assert trajectory.final_state == (60,)
    assert trajectory.total_reward == 2.0


def test_rollout_rank1_reward_terminal_success_stops_early() -> None:
    trajectory = rollout_rank1(
        initial_state=(60,),
        max_steps=3,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=2),
        active_sampler=ScriptedSampler(script=(1, 1, 1)),
        reward_fn=lambda context: TowerRewardResult(
            reward=10.0,
            is_terminal_success=True,
        ),
    )

    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].terminated
    assert not trajectory.steps[0].truncated


def test_rollout_rank1_goal_octave_window_allows_early_cadence_success() -> None:
    contexts = []

    def reward_fn(context: TowerRewardContext) -> TowerRewardResult:
        contexts.append(context)
        return TowerRewardResult(
            reward=10.0 if context.is_final_step else 0.0,
            is_terminal_success=context.is_final_step,
        )

    trajectory = rollout_rank1(
        initial_state=(48,),
        max_steps=8,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=7),
        active_sampler=ScriptedSampler(script=(7, 5, -7)),
        reward_fn=reward_fn,
        measure_size=2,
        context_measures=1,
        target_root_octave=4,
    )

    assert len(trajectory.steps) == 2
    assert contexts[0].is_final_step is False
    assert contexts[1].diagnostics["entered_goal_octave"] is True
    assert contexts[1].is_final_step is True
    assert contexts[1].diagnostics["goal_octave_entry_step_index"] == 1
    assert contexts[1].diagnostics["goal_octave_deadline_step_index"] == 4
    assert contexts[1].diagnostics["cadence_check_step"] is True
    assert trajectory.steps[-1].terminated
    assert not trajectory.steps[-1].truncated


def test_rollout_rank1_goal_octave_window_truncates_after_two_measures() -> None:
    contexts = []

    def reward_fn(context: TowerRewardContext) -> TowerRewardResult:
        contexts.append(context)
        return TowerRewardResult(reward=0.0)

    trajectory = rollout_rank1(
        initial_state=(48,),
        max_steps=8,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=6),
        active_sampler=ScriptedSampler(script=(6, 6, -1, 1, -1, 1, -1, 1)),
        reward_fn=reward_fn,
        measure_size=2,
        context_measures=1,
        target_root_octave=4,
    )

    assert len(trajectory.steps) == 5
    assert [context.is_final_step for context in contexts] == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert trajectory.steps[-1].step_index == 4
    assert trajectory.steps[1].diagnostics["entered_goal_octave"] is True
    assert trajectory.steps[-1].diagnostics["goal_octave_entry_step_index"] == 1
    assert trajectory.steps[-1].diagnostics["goal_octave_deadline_step_index"] == 4
    assert trajectory.steps[-1].diagnostics["goal_octave_window_expired"] is True
    assert not trajectory.steps[-1].terminated
    assert trajectory.steps[-1].truncated


def test_rollout_rank1_rejects_invalid_active_choice() -> None:
    with pytest.raises(ValueError, match="active choice must be legal for rank 1"):
        rollout_rank1(
            initial_state=(60,),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=1, max_step_size=2),
            active_sampler=ScriptedSampler(script=(3,)),
            reward_fn=lambda context: TowerRewardResult(reward=0.0),
        )


def test_rollout_rank1_rejects_non_rank_1_spec() -> None:
    with pytest.raises(ValueError, match="rank-1 graph spec"):
        rollout_rank1(
            initial_state=(60,),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=2),
            active_sampler=ScriptedSampler(script=(1,)),
            reward_fn=lambda context: TowerRewardResult(reward=0.0),
        )


def test_rollout_rank1_trajectory_loss_backpropagates_to_active_logprob() -> None:
    active_logprob = torch.tensor(-0.25, requires_grad=True)
    trajectory = rollout_rank1(
        initial_state=(60,),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=2),
        active_sampler=ScriptedSampler(script=(scripted_result(1, logprob=active_logprob),)),
        reward_fn=lambda context: TowerRewardResult(reward=2.0),
    )

    loss = policy_gradient_loss(trajectory).loss
    loss.backward()

    assert torch.allclose(active_logprob.grad, torch.tensor(-2.0))


def test_rollout_rank2_one_step_happy_path() -> None:
    rewards = []

    def reward_fn(context: TowerRewardContext) -> TowerRewardResult:
        rewards.append(context)
        return TowerRewardResult(reward=1.5, diagnostics={"kind": "reward"})

    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((1,),)),
        active_sampler=ScriptedSampler(script=(1,)),
        reward_fn=reward_fn,
        measure_size=4,
        context_measures=1,
    )

    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.rank == 2
    assert step.step_index == 0
    assert step.source_state == (60, 64)
    assert step.parent_state == (60,)
    assert step.parent_action == (1,)
    assert step.active_choice == 1
    assert step.assembled_action == (1, 1)
    assert step.attempted_target_state == (61, 65)
    assert step.realized_next_state == (61, 65)
    assert step.reward.reward == 1.5
    assert step.outcome == TRAJECTORY_OUTCOME_VALID
    assert not step.terminated
    assert step.truncated
    assert trajectory.final_state == (61, 65)
    assert rewards[0].source == (60, 64)
    assert rewards[0].target == (61, 65)
    assert rewards[0].action == (1, 1)


def test_rollout_rank2_parent_sampler_called_before_active_sampler() -> None:
    calls = []

    def parent_sampler(**kwargs: object) -> SamplerResult[tuple[int, ...]]:
        calls.append(("parent", kwargs["state"]))
        return SamplerResult(choice=(1,))

    def active_sampler(**kwargs: object) -> SamplerResult[int]:
        calls.append(("active", kwargs["active_choices"]))
        return SamplerResult(choice=1)

    rollout_rank2(
        initial_state=(60, 64),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=parent_sampler,
        active_sampler=active_sampler,
        reward_fn=lambda context: TowerRewardResult(reward=0.0),
    )

    assert calls == [
        ("parent", (60,)),
        ("active", (0, 1, 2)),
    ]


def test_rollout_rank2_multi_step_advances_history_and_final_state() -> None:
    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=2,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((1,), (1,))),
        active_sampler=ScriptedSampler(script=(2, -1)),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
    )

    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].source_state == (60, 64)
    assert trajectory.steps[0].realized_next_state == (61, 66)
    assert trajectory.steps[1].source_state == (61, 66)
    assert trajectory.steps[1].realized_next_state == (62, 65)
    assert trajectory.final_state == (62, 65)
    assert trajectory.total_reward == 2.0


def test_rollout_rank2_active_sampler_receives_active_choices_only() -> None:
    observed = []

    def active_sampler(**kwargs: object) -> SamplerResult[int]:
        observed.append(kwargs)
        return SamplerResult(choice=2)

    rollout_rank2(
        initial_state=(60, 64),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((1,),)),
        active_sampler=active_sampler,
        reward_fn=lambda context: TowerRewardResult(reward=0.0),
    )

    assert observed[0]["active_choices"] == (0, 1, 2)
    assert "lift_fiber" not in observed[0]


def test_rollout_rank2_keeps_parent_and_active_logprobs_separate() -> None:
    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(
            script=(scripted_result((1,), logprob=-5.0),),
        ),
        active_sampler=ScriptedSampler(
            script=(scripted_result(1, logprob=-0.25),),
        ),
        reward_fn=lambda context: TowerRewardResult(reward=0.0),
    )

    step = trajectory.steps[0]
    assert step.parent_logprob == -5.0
    assert step.active_logprob == -0.25


def test_rollout_rank2_stores_reward_callback_result() -> None:
    reward = TowerRewardResult(
        reward=2.0,
        hard_violation=False,
        is_terminal_success=True,
        diagnostics={"done": True},
    )

    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=2,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((1,),)),
        active_sampler=ScriptedSampler(script=(1,)),
        reward_fn=lambda context: reward,
    )

    step = trajectory.steps[0]
    assert step.reward is reward
    assert step.terminated
    assert not step.truncated
    assert len(trajectory.steps) == 1


def test_rollout_rank2_rejects_non_rank_2_spec() -> None:
    with pytest.raises(ValueError, match="rank-2 graph spec"):
        rollout_rank2(
            initial_state=(60, 64),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=1),
            parent_sampler=ScriptedSampler(script=((1,),)),
            active_sampler=ScriptedSampler(script=(1,)),
            reward_fn=lambda context: TowerRewardResult(reward=0.0),
        )


def test_rollout_rank2_records_invalid_extension_no_op_step() -> None:
    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((1,),)),
        active_sampler=ScriptedSampler(script=(3,)),
        reward_fn=lambda context: TowerRewardResult(reward=9.0),
    )

    step = trajectory.steps[0]
    assert step.outcome == TRAJECTORY_OUTCOME_INVALID_EXTENSION
    assert step.parent_action == (1,)
    assert step.active_choice == 3
    assert step.assembled_action == (1, 3)
    assert step.attempted_target_state == (61, 67)
    assert step.realized_next_state == step.source_state
    assert step.reward.reward == 0.0
    assert step.reward.diagnostics == {"outcome": TRAJECTORY_OUTCOME_INVALID_EXTENSION}
    assert not step.terminated
    assert step.truncated
    assert trajectory.final_state == (60, 64)


def test_rollout_rank2_invalid_extension_advances_time_with_repeated_state() -> None:
    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=2,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((1,), (1,))),
        active_sampler=ScriptedSampler(script=(3, 1)),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        measure_size=4,
        context_measures=1,
    )

    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].outcome == TRAJECTORY_OUTCOME_INVALID_EXTENSION
    assert trajectory.steps[1].outcome == TRAJECTORY_OUTCOME_VALID
    assert trajectory.steps[1].step_index == 1
    assert trajectory.steps[1].source_state == (60, 64)
    assert trajectory.steps[1].window.states[-2:] == ((60, 64), (60, 64))
    assert trajectory.steps[1].window.valid_mask[-2:] == (True, True)
    assert trajectory.final_state == (61, 65)


def test_rollout_rank2_records_empty_lift_fiber_without_active_sampler() -> None:
    active_calls = []

    def active_sampler(**kwargs: object) -> SamplerResult[int]:
        active_calls.append(kwargs)
        return SamplerResult(choice=1)

    trajectory = rollout_rank2(
        initial_state=(60, 63),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, pitch_max=63, max_step_size=1),
        parent_sampler=ScriptedSampler(script=((1,),)),
        active_sampler=active_sampler,
        reward_fn=lambda context: TowerRewardResult(reward=9.0),
    )

    step = trajectory.steps[0]
    assert active_calls == []
    assert step.outcome == TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER
    assert step.parent_action == (1,)
    assert step.active_choice is None
    assert step.assembled_action == (0, 0)
    assert step.attempted_target_state == (60, 63)
    assert step.realized_next_state == step.source_state
    assert step.reward.reward == 0.0
    assert step.reward.diagnostics == {"outcome": TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER}
    assert step.diagnostics["empty_lift_fiber"] is True
    assert "parent_failure" not in step.diagnostics


def test_rollout_rank2_empty_lift_fiber_advances_time() -> None:
    trajectory = rollout_rank2(
        initial_state=(60, 63),
        max_steps=2,
        graph_spec=TowerGraphSpec(rank=2, pitch_max=63, max_step_size=1),
        parent_sampler=ScriptedSampler(script=((1,), (1,))),
        active_sampler=ScriptedSampler(script=(1,)),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        measure_size=4,
        context_measures=1,
    )

    assert len(trajectory.steps) == 2
    assert all(
        step.outcome == TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER
        for step in trajectory.steps
    )
    assert trajectory.steps[1].step_index == 1
    assert trajectory.steps[1].window.states[-2:] == ((60, 63), (60, 63))


def test_rollout_rank2_records_parent_failure_and_truncates() -> None:
    active_calls = []

    def active_sampler(**kwargs: object) -> SamplerResult[int]:
        active_calls.append(kwargs)
        return SamplerResult(choice=1)

    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=2,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((0,),)),
        active_sampler=active_sampler,
        reward_fn=lambda context: TowerRewardResult(reward=9.0),
    )

    assert active_calls == []
    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.outcome == TRAJECTORY_OUTCOME_PARENT_FAILURE
    assert step.parent_action == (0,)
    assert step.active_choice is None
    assert step.realized_next_state == step.source_state
    assert step.truncated
    assert not step.terminated
    assert step.reward.reward == 0.0
    assert step.reward.diagnostics == {"outcome": TRAJECTORY_OUTCOME_PARENT_FAILURE}
    assert step.diagnostics["parent_failure"] is True
    assert "invalid_extension" not in step.diagnostics


def test_rollout_rank2_projected_parent_data_recoverable_on_demand() -> None:
    trajectory = rollout_rank2(
        initial_state=(60, 64),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=2),
        parent_sampler=ScriptedSampler(script=((1,),)),
        active_sampler=ScriptedSampler(script=(-1,)),
        reward_fn=lambda context: TowerRewardResult(reward=0.0),
        measure_size=4,
        context_measures=1,
    )

    step = trajectory.steps[0]
    parent_window = project_window(step.window)

    assert project_action(step.assembled_action) == step.parent_action
    assert parent_window.states[-1] == step.parent_state
    assert parent_window.valid_mask == step.window.valid_mask
    assert parent_window.bar_positions == step.window.bar_positions
