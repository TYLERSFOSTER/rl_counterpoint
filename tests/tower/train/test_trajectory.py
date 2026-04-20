"""Tests for tower trajectory record contracts."""

from __future__ import annotations

import pytest
import torch

from tower.reward.result import TowerRewardResult
from tower.train.trajectory import (
    TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER,
    TRAJECTORY_OUTCOME_INVALID_EXTENSION,
    TRAJECTORY_OUTCOME_PARENT_FAILURE,
    TRAJECTORY_OUTCOME_VALID,
    TowerTrajectory,
    TowerTrajectoryStep,
)
from tower.window import TowerWindow, build_window


def make_rank_2_window() -> TowerWindow:
    return build_window(
        history=((60, 64),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )


def make_step(
    *,
    source_state: tuple[int, int] = (60, 64),
    attempted_target_state: tuple[int, int] = (61, 65),
    realized_next_state: tuple[int, int] = (61, 65),
    reward: TowerRewardResult | None = None,
    outcome: str = TRAJECTORY_OUTCOME_VALID,
) -> TowerTrajectoryStep:
    return TowerTrajectoryStep(
        rank=2,
        step_index=0,
        source_state=source_state,
        window=make_rank_2_window(),
        parent_state=(60,),
        parent_action=(1,),
        active_choice=1,
        assembled_action=(1, 1),
        attempted_target_state=attempted_target_state,
        realized_next_state=realized_next_state,
        active_logprob=None,
        parent_logprob=None,
        reward=reward
        if reward is not None
        else TowerRewardResult(reward=1.25, diagnostics={"kind": "test"}),
        terminated=False,
        truncated=False,
        outcome=outcome,  # type: ignore[arg-type]
        diagnostics={"source": "unit_test"},
    )


def test_step_records_required_rank_2_fields() -> None:
    step = make_step()

    assert step.rank == 2
    assert step.source_state == (60, 64)
    assert step.parent_state == (60,)
    assert step.parent_action == (1,)
    assert step.active_choice == 1
    assert step.assembled_action == (1, 1)
    assert step.attempted_target_state == (61, 65)
    assert step.realized_next_state == (61, 65)
    assert step.outcome == TRAJECTORY_OUTCOME_VALID
    assert step.diagnostics == {"source": "unit_test"}


def test_step_stores_reward_result_not_scalar() -> None:
    reward = TowerRewardResult(
        reward=-0.5,
        hard_violation=True,
        is_terminal_success=False,
        diagnostics={"outcome": "hard"},
    )

    step = make_step(reward=reward)

    assert step.reward is reward
    assert step.reward.hard_violation is True
    assert step.reward.diagnostics == {"outcome": "hard"}


def test_step_allows_masked_padding_state_in_window() -> None:
    window = TowerWindow(
        states=((0, 0), (60, 64)),
        bar_positions=(-1, 0),
        valid_mask=(False, True),
    )

    step = TowerTrajectoryStep(
        rank=2,
        step_index=0,
        source_state=(60, 64),
        window=window,
        parent_state=(60,),
        parent_action=(1,),
        active_choice=1,
        assembled_action=(1, 1),
        attempted_target_state=(61, 65),
        realized_next_state=(61, 65),
        active_logprob=None,
        parent_logprob=None,
        reward=TowerRewardResult(reward=0.0),
        terminated=False,
        truncated=False,
        outcome=TRAJECTORY_OUTCOME_VALID,
    )

    assert step.window == window


def test_step_rejects_valid_window_state_rank_mismatch() -> None:
    window = TowerWindow(
        states=((0, 0), (60,)),
        bar_positions=(-1, 0),
        valid_mask=(False, True),
    )

    with pytest.raises(ValueError, match="state length must match rank"):
        TowerTrajectoryStep(
            rank=2,
            step_index=0,
            source_state=(60, 64),
            window=window,
            parent_state=(60,),
            parent_action=(1,),
            active_choice=1,
            assembled_action=(1, 1),
            attempted_target_state=(61, 65),
            realized_next_state=(61, 65),
            active_logprob=None,
            parent_logprob=None,
            reward=TowerRewardResult(reward=0.0),
            terminated=False,
            truncated=False,
            outcome=TRAJECTORY_OUTCOME_VALID,
        )


def test_step_requires_parent_fields_for_rank_greater_than_one() -> None:
    with pytest.raises(ValueError, match="parent_state is required"):
        TowerTrajectoryStep(
            rank=2,
            step_index=0,
            source_state=(60, 64),
            window=make_rank_2_window(),
            parent_state=None,
            parent_action=(1,),
            active_choice=1,
            assembled_action=(1, 1),
            attempted_target_state=(61, 65),
            realized_next_state=(61, 65),
            active_logprob=None,
            parent_logprob=None,
            reward=TowerRewardResult(reward=0.0),
            terminated=False,
            truncated=False,
            outcome=TRAJECTORY_OUTCOME_VALID,
        )


def test_step_distinguishes_attempted_and_realized_invalid_extension() -> None:
    step = make_step(
        attempted_target_state=(61, 72),
        realized_next_state=(60, 64),
        reward=TowerRewardResult(
            reward=0.0,
            diagnostics={"outcome": TRAJECTORY_OUTCOME_INVALID_EXTENSION},
        ),
        outcome=TRAJECTORY_OUTCOME_INVALID_EXTENSION,
    )

    assert step.attempted_target_state != step.realized_next_state
    assert step.realized_next_state == step.source_state
    assert step.reward.diagnostics["outcome"] == TRAJECTORY_OUTCOME_INVALID_EXTENSION


def test_step_accepts_slice_4_outcome_labels() -> None:
    assert make_step(outcome=TRAJECTORY_OUTCOME_VALID).outcome == TRAJECTORY_OUTCOME_VALID
    assert (
        make_step(outcome=TRAJECTORY_OUTCOME_INVALID_EXTENSION).outcome
        == TRAJECTORY_OUTCOME_INVALID_EXTENSION
    )
    assert (
        make_step(outcome=TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER).outcome
        == TRAJECTORY_OUTCOME_EMPTY_LIFT_FIBER
    )
    assert (
        make_step(outcome=TRAJECTORY_OUTCOME_PARENT_FAILURE).outcome
        == TRAJECTORY_OUTCOME_PARENT_FAILURE
    )


def test_step_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="outcome is not a recognized"):
        make_step(outcome="surprise")


def test_step_accepts_scalar_tensor_logprobs() -> None:
    active_logprob = torch.tensor(-0.5, requires_grad=True)
    parent_logprob = torch.tensor(-3.0)

    step = TowerTrajectoryStep(
        rank=2,
        step_index=0,
        source_state=(60, 64),
        window=make_rank_2_window(),
        parent_state=(60,),
        parent_action=(1,),
        active_choice=1,
        assembled_action=(1, 1),
        attempted_target_state=(61, 65),
        realized_next_state=(61, 65),
        active_logprob=active_logprob,
        parent_logprob=parent_logprob,
        reward=TowerRewardResult(reward=0.0),
        terminated=False,
        truncated=False,
        outcome=TRAJECTORY_OUTCOME_VALID,
    )

    assert step.active_logprob is active_logprob
    assert step.parent_logprob is parent_logprob


def test_step_rejects_vector_tensor_logprob() -> None:
    with pytest.raises(ValueError, match="active_logprob tensor must be scalar"):
        TowerTrajectoryStep(
            rank=2,
            step_index=0,
            source_state=(60, 64),
            window=make_rank_2_window(),
            parent_state=(60,),
            parent_action=(1,),
            active_choice=1,
            assembled_action=(1, 1),
            attempted_target_state=(61, 65),
            realized_next_state=(61, 65),
            active_logprob=torch.tensor([-0.5, -0.25]),
            parent_logprob=None,
            reward=TowerRewardResult(reward=0.0),
            terminated=False,
            truncated=False,
            outcome=TRAJECTORY_OUTCOME_VALID,
        )


def test_trajectory_exposes_rank_states_and_total_reward() -> None:
    step_0 = make_step(
        source_state=(60, 64),
        attempted_target_state=(61, 65),
        realized_next_state=(61, 65),
        reward=TowerRewardResult(reward=1.25),
    )
    step_1 = make_step(
        source_state=(61, 65),
        attempted_target_state=(62, 66),
        realized_next_state=(62, 66),
        reward=TowerRewardResult(reward=-0.25),
    )

    trajectory = TowerTrajectory(steps=(step_0, step_1))

    assert trajectory.rank == 2
    assert trajectory.initial_state == (60, 64)
    assert trajectory.final_state == (62, 66)
    assert trajectory.total_reward == 1.0


def test_trajectory_rejects_mixed_ranks() -> None:
    rank_1_step = TowerTrajectoryStep(
        rank=1,
        step_index=0,
        source_state=(60,),
        window=build_window(
            history=((60,),),
            step_index=0,
            measure_size=4,
            context_measures=1,
        ),
        parent_state=None,
        parent_action=None,
        active_choice=1,
        assembled_action=(1,),
        attempted_target_state=(61,),
        realized_next_state=(61,),
        active_logprob=None,
        parent_logprob=None,
        reward=TowerRewardResult(reward=0.0),
        terminated=False,
        truncated=False,
        outcome=TRAJECTORY_OUTCOME_VALID,
    )

    with pytest.raises(ValueError, match="same rank"):
        TowerTrajectory(steps=(make_step(), rank_1_step))
