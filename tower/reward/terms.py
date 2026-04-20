"""Composable reward term contracts for the tower model."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Protocol

from tower.reward.context import TowerRewardContext
from tower.reward.result import TowerRewardResult
from tower.reward.success import SuccessResult


class RewardTerm(Protocol):
    """Callable reward term over a tower reward context."""

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        """Evaluate one reward term."""


@dataclass(frozen=True)
class CompositeRewardTerm:
    """Evaluate reward terms in order and combine their structured results."""

    terms: tuple[RewardTerm, ...]
    diagnostics_key: str = "terms"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.terms, tuple):
            raise TypeError("terms must be a tuple")

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        child_results = []
        for index, term in enumerate(self.terms):
            result = term(context)
            if not isinstance(result, TowerRewardResult):
                raise TypeError("reward terms must return TowerRewardResult")
            child_results.append((index, result))

        return TowerRewardResult(
            reward=sum(result.reward for _, result in child_results),
            hard_violation=any(result.hard_violation for _, result in child_results),
            is_terminal_success=any(
                result.is_terminal_success for _, result in child_results
            ),
            diagnostics={
                **self.diagnostics,
                self.diagnostics_key: tuple(
                    {
                        "index": index,
                        "reward": result.reward,
                        "hard_violation": result.hard_violation,
                        "is_terminal_success": result.is_terminal_success,
                        "diagnostics": result.diagnostics,
                    }
                    for index, result in child_results
                ),
            },
        )


@dataclass(frozen=True)
class SuccessRewardTerm:
    """Adapt a success predicate into a structured reward term."""

    predicate: Callable[[TowerRewardContext], SuccessResult]
    success_reward: float
    failure_reward: float = 0.0
    diagnostics_key: str = "success"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        result = self.predicate(context)
        if not isinstance(result, SuccessResult):
            raise TypeError("success predicate must return SuccessResult")

        return TowerRewardResult(
            reward=self.success_reward if result.success else self.failure_reward,
            hard_violation=False,
            is_terminal_success=result.success,
            diagnostics={
                **self.diagnostics,
                self.diagnostics_key: result.diagnostics,
            },
        )
