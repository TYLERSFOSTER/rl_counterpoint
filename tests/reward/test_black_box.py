"""Tests for temporary black-box reward implementations.

These tests cover placeholder rewards used before the TC21M evaluator exists.
They verify that the placeholder obeys the reward protocol and exposes enough
diagnostic information for environment plumbing and smoke tests.
"""

from __future__ import annotations

from rl_counterpoint.reward.black_box import ConstantReward
from rl_counterpoint.reward.protocol import RewardContext, RewardResult


def test_constant_reward_returns_reward_result() -> None:
    """ConstantReward returns the structured RewardResult type."""
    reward_fn = ConstantReward(reward=1.25)

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=0))

    assert isinstance(result, RewardResult)


def test_constant_reward_returns_configured_reward() -> None:
    """ConstantReward returns the fixed scalar value it was configured with."""
    reward_fn = ConstantReward(reward=-0.5)

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=0))

    assert result.reward == -0.5
    assert not result.hard_violation
    assert not result.is_terminal_success


def test_constant_reward_includes_transition_diagnostics() -> None:
    """ConstantReward records source, target, step index, and placeholder kind."""
    reward_fn = ConstantReward(reward=0.0)

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=5))

    assert result.diagnostics["kind"] == "constant"
    assert result.diagnostics["source"] == (3, 6)
    assert result.diagnostics["target"] == (4, 7)
    assert result.diagnostics["step_index"] == 5


def test_constant_reward_merges_custom_diagnostics() -> None:
    """Custom diagnostics are included in the result diagnostics mapping."""
    reward_fn = ConstantReward(
        reward=0.0,
        diagnostics={"note": "plumbing-only", "kind": "custom"},
    )

    result = reward_fn((3, 6), (4, 7), RewardContext(step_index=0))

    assert result.diagnostics["kind"] == "custom"
    assert result.diagnostics["note"] == "plumbing-only"
    assert result.diagnostics["source"] == (3, 6)
    assert result.diagnostics["target"] == (4, 7)
