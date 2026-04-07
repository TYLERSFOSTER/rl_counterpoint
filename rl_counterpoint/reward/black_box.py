"""Temporary black-box reward implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from rl_counterpoint.graph.state_space import ChordState
from rl_counterpoint.reward.protocol import RewardContext, RewardResult


@dataclass(frozen=True)
class ConstantReward:
    """Return a fixed reward for every transition.

    This is a plumbing reward, not a music-theory reward. It exists so the
    environment and learner can depend on the reward protocol before the TC21M
    evaluator is formalized.
    """

    reward: float = 0.0
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __call__(
        self,
        source: ChordState,
        target: ChordState,
        context: RewardContext,
    ) -> RewardResult:
        return RewardResult(
            reward=self.reward,
            diagnostics={
                "kind": "constant",
                "source": source,
                "target": target,
                "step_index": context.step_index,
                **self.diagnostics,
            },
        )
