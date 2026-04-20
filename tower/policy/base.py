"""Policy contracts for trainable tower ranks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import torch

from tower.state_action import TowerState
from tower.window import TowerWindow


@dataclass(frozen=True)
class PolicyOutput:
    """Policy logits plus diagnostic metadata."""

    logits: torch.Tensor
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")


class RankPolicy(Protocol):
    """Protocol for one trainable rank-local policy."""

    rank: int

    def __call__(
        self,
        *,
        state: TowerState,
        window: TowerWindow,
    ) -> PolicyOutput:
        """Return policy logits for one rank-local decision point."""


def freeze_parent_policy(policy: torch.nn.Module) -> torch.nn.Module:
    """Freeze a parent policy module for child-rank training."""
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def policy_parameters_frozen(policy: torch.nn.Module) -> bool:
    """Return True iff all policy parameters are frozen."""
    return all(not parameter.requires_grad for parameter in policy.parameters())
