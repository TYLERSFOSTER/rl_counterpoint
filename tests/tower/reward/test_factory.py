"""Tests for tower reward factory helpers."""

from __future__ import annotations

import pytest

from tower.reward.context import TowerRewardContext
from tower.reward.factory import (
    Rank1RewardFactoryConfig,
    Rank1RewardFunction,
    build_rank1_reward_fn,
)
from tower.reward.result import TowerRewardResult
from tower.window import build_window


def make_context(
    *,
    history: tuple[tuple[int, ...], ...] = ((67,), (60,)),
    action: tuple[int, ...] = (2,),
    step_index: int = 3,
    measure_size: int = 4,
    key_pitch_class: int | None = None,
    is_final_step: bool = True,
    rank: int = 1,
) -> TowerRewardContext:
    return TowerRewardContext(
        rank=rank,
        step_index=step_index,
        source=history[-1],
        target=tuple(
            pitch + delta for pitch, delta in zip(history[-1], action, strict=True)
        ),
        action=action,
        window=build_window(
            history=history,
            step_index=step_index,
            measure_size=measure_size,
            context_measures=2,
        ),
        measure_size=measure_size,
        key_pitch_class=key_pitch_class,
        is_final_step=is_final_step,
    )


def test_build_rank1_reward_fn_returns_reward_result() -> None:
    reward_fn = build_rank1_reward_fn()

    result = reward_fn(make_context())

    assert isinstance(reward_fn, Rank1RewardFunction)
    assert isinstance(result, TowerRewardResult)


def test_rank1_reward_factory_combines_cadence_and_melody_terms() -> None:
    reward_fn = build_rank1_reward_fn(
        terminal_cadence_reward=10.0,
        recovery_reward=0.75,
    )

    result = reward_fn(make_context(history=((67,), (60,)), action=(2,)))

    assert result.reward == 10.75
    assert result.is_terminal_success is True
    child_results = result.diagnostics["terms"]
    assert child_results[0]["reward"] == 10.0
    assert child_results[1]["reward"] == 0.0
    assert child_results[2]["reward"] == 0.75


def test_rank1_reward_factory_injects_configured_key() -> None:
    reward_fn = build_rank1_reward_fn(key_pitch_class=2)

    result = reward_fn(
        make_context(
            history=((69,), (62,)),
            key_pitch_class=None,
        )
    )

    cadence_diagnostics = result.diagnostics["terms"][0]["diagnostics"]["cadence"]
    assert result.is_terminal_success is True
    assert cadence_diagnostics["dominant_pitch_class"] == 9
    assert cadence_diagnostics["tonic_pitch_class"] == 2
    assert cadence_diagnostics["reason"] == "success"


def test_rank1_reward_factory_overrides_context_key_with_configured_key() -> None:
    reward_fn = build_rank1_reward_fn(key_pitch_class=0)

    result = reward_fn(
        make_context(
            history=((67,), (60,)),
            key_pitch_class=5,
        )
    )

    cadence_diagnostics = result.diagnostics["terms"][0]["diagnostics"]["cadence"]
    assert result.is_terminal_success is True
    assert cadence_diagnostics["tonic_pitch_class"] == 0


def test_rank1_reward_factory_preserves_cadence_failure_diagnostics() -> None:
    reward_fn = build_rank1_reward_fn(cadence_failure_reward=-2.0)

    result = reward_fn(make_context(history=((65,), (60,)), action=(1,)))

    cadence_result = result.diagnostics["terms"][0]
    assert result.is_terminal_success is False
    assert cadence_result["reward"] == -2.0
    assert cadence_result["diagnostics"]["cadence"]["reason"] == "wrong_root_motion"


def test_rank1_reward_factory_exposes_top_level_diagnostics() -> None:
    reward_fn = build_rank1_reward_fn(key_pitch_class=3)

    result = reward_fn(make_context(history=((70,), (63,))))

    assert result.diagnostics["kind"] == "rank1_reward"
    assert result.diagnostics["key_pitch_class"] == 3
    assert len(result.diagnostics["terms"]) == 3


def test_rank1_reward_factory_rejects_non_rank_1_context() -> None:
    reward_fn = build_rank1_reward_fn()
    context = make_context(
        history=((60, 64), (62, 65)),
        action=(1, 1),
        rank=2,
    )

    with pytest.raises(ValueError, match="requires rank 1 context"):
        reward_fn(context)


def test_rank1_reward_factory_validates_config_values() -> None:
    with pytest.raises(ValueError, match="key_pitch_class must be in"):
        Rank1RewardFactoryConfig(key_pitch_class=12)

    with pytest.raises(TypeError, match="terminal_cadence_reward must be"):
        Rank1RewardFactoryConfig(terminal_cadence_reward=True)

    with pytest.raises(ValueError, match="max_recent_range must be non-negative"):
        build_rank1_reward_fn(max_recent_range=-1)

    with pytest.raises(ValueError, match="large_leap_threshold must be at least 1"):
        build_rank1_reward_fn(large_leap_threshold=0)
