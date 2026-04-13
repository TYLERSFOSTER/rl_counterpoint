"""Reward interface contract for counterpoint environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from rl_counterpoint.envs.observation import TimedChordWindow
from rl_counterpoint.graph.actions import StepDelta
from rl_counterpoint.graph.state_space import ChordState


@dataclass(frozen=True)
class RewardContext:
    """Episode context supplied to reward functions.

    This carries non-local information without making reward evaluators depend
    directly on the environment object.
    """

    step_index: int
    max_steps: int | None = None
    max_step_size: int | None = None
    measure_size: int | None = None
    history: tuple[ChordState, ...] = ()
    step_delta: StepDelta | None = None
    key_pitch_class: int | None = None
    timed_chord_window: TimedChordWindow | None = None
    target_root_octave: int | None = None
    is_final_step: bool = False


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
