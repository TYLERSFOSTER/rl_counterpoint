"""Reward factory helpers for tower training."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Real

from tower.reward.context import TowerRewardContext
from tower.reward.melody import LargeLeapRecoveryTerm, RecentMelodicRangePenalty
from tower.reward.result import TowerRewardResult
from tower.reward.success import rank1_projected_cadence_success
from tower.reward.terms import CompositeRewardTerm, RewardTerm, SuccessRewardTerm


@dataclass(frozen=True)
class Rank1RewardFactoryConfig:
    """Configuration for the first rank-1 musical reward bundle."""

    key_pitch_class: int = 0
    terminal_cadence_reward: float = 10.0
    cadence_failure_reward: float = 0.0
    max_recent_range: int = 12
    range_penalty: float = -1.0
    large_leap_threshold: int = 6
    recovery_step_threshold: int = 3
    recovery_reward: float = 0.5
    failure_penalty: float = -0.5

    def __post_init__(self) -> None:
        _validate_pitch_class(self.key_pitch_class)
        _validate_number(
            self.terminal_cadence_reward,
            field_name="terminal_cadence_reward",
        )
        _validate_number(
            self.cadence_failure_reward,
            field_name="cadence_failure_reward",
        )


@dataclass(frozen=True)
class Rank1RewardFunction:
    """Callable rank-1 reward bundle built from composable reward terms."""

    config: Rank1RewardFactoryConfig
    term: RewardTerm = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "term",
            CompositeRewardTerm(
                terms=(
                    SuccessRewardTerm(
                        predicate=rank1_projected_cadence_success,
                        success_reward=float(self.config.terminal_cadence_reward),
                        failure_reward=float(self.config.cadence_failure_reward),
                        diagnostics_key="cadence",
                    ),
                    RecentMelodicRangePenalty(
                        max_recent_range=self.config.max_recent_range,
                        penalty=float(self.config.range_penalty),
                    ),
                    LargeLeapRecoveryTerm(
                        large_leap_threshold=self.config.large_leap_threshold,
                        recovery_step_threshold=self.config.recovery_step_threshold,
                        recovery_reward=float(self.config.recovery_reward),
                        failure_penalty=float(self.config.failure_penalty),
                    ),
                ),
                diagnostics={
                    "kind": "rank1_reward",
                    "key_pitch_class": self.config.key_pitch_class,
                },
            ),
        )

    def __call__(self, context: TowerRewardContext) -> TowerRewardResult:
        if not isinstance(context, TowerRewardContext):
            raise TypeError("context must be a TowerRewardContext")
        if context.rank != 1:
            raise ValueError("rank-1 reward function requires rank 1 context")

        keyed_context = replace(
            context,
            key_pitch_class=self.config.key_pitch_class,
        )
        return self.term(keyed_context)


def build_rank1_reward_fn(
    *,
    key_pitch_class: int = 0,
    terminal_cadence_reward: float = 10.0,
    cadence_failure_reward: float = 0.0,
    max_recent_range: int = 12,
    range_penalty: float = -1.0,
    large_leap_threshold: int = 6,
    recovery_step_threshold: int = 3,
    recovery_reward: float = 0.5,
    failure_penalty: float = -0.5,
) -> Rank1RewardFunction:
    """Build the first rank-1 cadence and melodic-shape reward function."""
    return Rank1RewardFunction(
        config=Rank1RewardFactoryConfig(
            key_pitch_class=key_pitch_class,
            terminal_cadence_reward=terminal_cadence_reward,
            cadence_failure_reward=cadence_failure_reward,
            max_recent_range=max_recent_range,
            range_penalty=range_penalty,
            large_leap_threshold=large_leap_threshold,
            recovery_step_threshold=recovery_step_threshold,
            recovery_reward=recovery_reward,
            failure_penalty=failure_penalty,
        )
    )


def _validate_pitch_class(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("key_pitch_class must be an integer")
    if value < 0 or value > 11:
        raise ValueError("key_pitch_class must be in [0, 11]")


def _validate_number(value: float, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
