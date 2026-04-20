"""Tests for tower reward result shell."""

from __future__ import annotations

from tower.reward.result import TowerRewardResult


def test_reward_result_default_flags() -> None:
    result = TowerRewardResult(reward=1.5)

    assert result.reward == 1.5
    assert result.hard_violation is False
    assert result.is_terminal_success is False
    assert result.diagnostics == {}


def test_reward_result_accepts_diagnostics() -> None:
    diagnostics = {"kind": "example", "rank": 1}

    result = TowerRewardResult(reward=0.0, diagnostics=diagnostics)

    assert result.diagnostics == diagnostics
