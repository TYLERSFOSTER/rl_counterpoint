"""Reward interface contract for counterpoint environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from rl_counterpoint.graph.state_space import ChordState


@dataclass(frozen=True)
class RewardContext:
    """Episode context supplied to reward functions.

    This carries non-local information without making reward evaluators depend
    directly on the environment object.
    """

    step_index: int
    max_steps: int | None = None
    measure_size: int | None = None
    history: tuple[ChordState, ...] = ()


@dataclass(frozen=True)
class RewardResult:
    """Structured reward output consumed by the environment."""

    reward: float
    hard_violation: bool = False
    is_terminal_success: bool = False
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class RewardFn(Protocol):
    """Callable contract for any reward implementation."""

    def __call__(
        self,
        source: ChordState,
        target: ChordState,
        context: RewardContext,
    ) -> RewardResult:
        """Score a transition from source to target."""
