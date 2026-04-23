"""Policy-gradient loss helpers for tower training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from tower.train.trajectory import LogProb, TowerTrajectory


@dataclass(frozen=True)
class PolicyGradientLossResult:
    """Structured policy-gradient loss output."""

    loss: torch.Tensor
    returns: torch.Tensor
    diagnostics: Mapping[str, object]


def discounted_returns(
    rewards: tuple[float, ...],
    *,
    gamma: float,
) -> torch.Tensor:
    """Compute reward-to-go returns."""
    _validate_gamma(gamma)
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def trajectory_returns(
    trajectory: TowerTrajectory,
    *,
    gamma: float,
) -> torch.Tensor:
    """Compute reward-to-go returns from a trajectory."""
    rewards = tuple(float(step.reward.reward) for step in trajectory.steps)
    return discounted_returns(rewards, gamma=gamma)


def policy_gradient_loss(
    trajectory: TowerTrajectory,
    *,
    gamma: float = 1.0,
    normalize_returns: bool = False,
) -> PolicyGradientLossResult:
    """Compute active-tier REINFORCE loss for one trajectory."""
    returns = trajectory_returns(trajectory, gamma=gamma)
    if normalize_returns and returns.numel() > 1:
        std = returns.std(unbiased=False)
        if torch.ne(std, torch.tensor(0.0)):
            returns = (returns - returns.mean()) / std

    active_step_indices = tuple(
        index
        for index, step in enumerate(trajectory.steps)
        if step.active_logprob is not None
    )
    active_logprobs = tuple(
        _logprob_to_tensor(
            trajectory.steps[index].active_logprob,
            step_index=trajectory.steps[index].step_index,
        )
        for index in active_step_indices
    )
    if not active_logprobs:
        loss = torch.tensor(0.0, dtype=torch.float32)
    else:
        logprob_tensor = torch.stack(active_logprobs)
        active_returns = returns[list(active_step_indices)].to(
            dtype=logprob_tensor.dtype,
            device=logprob_tensor.device,
        )
        loss = -(logprob_tensor * active_returns).sum()

    return PolicyGradientLossResult(
        loss=loss,
        returns=returns,
        diagnostics={
            "step_count": len(trajectory.steps),
            "active_step_count": len(active_logprobs),
            "gamma": gamma,
            "normalize_returns": normalize_returns,
        },
    )


def _validate_gamma(gamma: float) -> None:
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")


def _logprob_to_tensor(value: LogProb | None, *, step_index: int) -> torch.Tensor:
    if value is None:
        raise ValueError(f"step {step_index} missing active_logprob")
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"step {step_index} active_logprob tensor must be scalar")
        return value
    return torch.tensor(value, dtype=torch.float32)
