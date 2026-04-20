"""Tests for tower policy-gradient loss helpers."""

from __future__ import annotations

import pytest
import torch

from tower.reward.result import TowerRewardResult
from tower.train.losses import (
    discounted_returns,
    policy_gradient_loss,
    trajectory_returns,
)
from tower.train.trajectory import TowerTrajectory, TowerTrajectoryStep
from tower.window import build_window


def make_rank_1_step(
    *,
    step_index: int,
    reward: float,
    active_logprob: float | torch.Tensor | None,
    parent_logprob: float | torch.Tensor | None = None,
) -> TowerTrajectoryStep:
    source = (60 + step_index,)
    target = (61 + step_index,)
    return TowerTrajectoryStep(
        rank=1,
        step_index=step_index,
        source_state=source,
        window=build_window(
            history=(source,),
            step_index=step_index,
            measure_size=4,
            context_measures=1,
        ),
        parent_state=None,
        parent_action=None,
        active_choice=1,
        assembled_action=(1,),
        attempted_target_state=target,
        realized_next_state=target,
        active_logprob=active_logprob,
        parent_logprob=parent_logprob,
        reward=TowerRewardResult(reward=reward),
        terminated=False,
        truncated=False,
        outcome="valid",
    )


def make_trajectory(
    *,
    active_logprobs: tuple[float | torch.Tensor | None, ...],
    parent_logprobs: tuple[float | torch.Tensor | None, ...] | None = None,
) -> TowerTrajectory:
    rewards = (1.0, 2.0, 3.0)
    if parent_logprobs is None:
        parent_logprobs = (None,) * len(active_logprobs)

    return TowerTrajectory(
        steps=tuple(
            make_rank_1_step(
                step_index=index,
                reward=rewards[index],
                active_logprob=active_logprob,
                parent_logprob=parent_logprobs[index],
            )
            for index, active_logprob in enumerate(active_logprobs)
        )
    )


def test_discounted_returns_compute_reward_to_go() -> None:
    returns = discounted_returns((1.0, 2.0, 3.0), gamma=0.5)

    assert torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0]))


def test_discounted_returns_reject_invalid_gamma() -> None:
    with pytest.raises(ValueError, match="gamma must be in \\[0, 1\\]"):
        discounted_returns((1.0,), gamma=1.1)


def test_trajectory_returns_extract_rewards() -> None:
    trajectory = make_trajectory(active_logprobs=(-0.1, -0.2, -0.3))

    returns = trajectory_returns(trajectory, gamma=1.0)

    assert torch.allclose(returns, torch.tensor([6.0, 5.0, 3.0]))


def test_policy_gradient_loss_uses_active_logprobs_only() -> None:
    trajectory = make_trajectory(
        active_logprobs=(-0.1, -0.2, -0.3),
        parent_logprobs=(-10.0, -10.0, -10.0),
    )

    result = policy_gradient_loss(trajectory, gamma=1.0)

    expected = -(
        torch.tensor([-0.1, -0.2, -0.3])
        * torch.tensor([6.0, 5.0, 3.0])
    ).sum()
    assert torch.allclose(result.loss, expected)
    assert result.diagnostics["step_count"] == 3


def test_policy_gradient_loss_rejects_missing_active_logprob() -> None:
    trajectory = make_trajectory(active_logprobs=(-0.1, None, -0.3))

    with pytest.raises(ValueError, match="step 1 missing active_logprob"):
        policy_gradient_loss(trajectory)


def test_policy_gradient_loss_preserves_active_logprob_gradients_only() -> None:
    active_logprob_0 = torch.tensor(-0.1, requires_grad=True)
    active_logprob_1 = torch.tensor(-0.2, requires_grad=True)
    parent_logprob = torch.tensor(-10.0, requires_grad=True)
    trajectory = make_trajectory(
        active_logprobs=(active_logprob_0, active_logprob_1, -0.3),
        parent_logprobs=(parent_logprob, parent_logprob, parent_logprob),
    )

    result = policy_gradient_loss(trajectory, gamma=1.0)
    result.loss.backward()

    assert torch.allclose(active_logprob_0.grad, torch.tensor(-6.0))
    assert torch.allclose(active_logprob_1.grad, torch.tensor(-5.0))
    assert parent_logprob.grad is None


def test_policy_gradient_loss_can_normalize_returns() -> None:
    trajectory = make_trajectory(active_logprobs=(-0.1, -0.2, -0.3))

    result = policy_gradient_loss(
        trajectory,
        gamma=1.0,
        normalize_returns=True,
    )

    assert torch.isclose(result.returns.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(result.returns.std(unbiased=False), torch.tensor(1.0))

