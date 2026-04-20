"""Tests for tower reward term composition."""

from __future__ import annotations

import pytest

from tower.reward.context import TowerRewardContext
from tower.reward.result import TowerRewardResult
from tower.reward.success import SuccessResult
from tower.reward.terms import CompositeRewardTerm, SuccessRewardTerm
from tower.window import build_window


def make_context() -> TowerRewardContext:
    return TowerRewardContext(
        rank=1,
        step_index=0,
        source=(60,),
        target=(62,),
        action=(2,),
        window=build_window(
            history=((60,),),
            step_index=0,
            measure_size=4,
            context_measures=1,
        ),
    )


def test_function_term_returns_reward_result() -> None:
    def term(context: TowerRewardContext) -> TowerRewardResult:
        assert context.rank == 1
        return TowerRewardResult(reward=1.0, diagnostics={"term": "one"})

    result = term(make_context())

    assert result == TowerRewardResult(reward=1.0, diagnostics={"term": "one"})


def test_composite_sums_rewards() -> None:
    composite = CompositeRewardTerm(
        terms=(
            lambda context: TowerRewardResult(reward=1.25),
            lambda context: TowerRewardResult(reward=-0.5),
        )
    )

    result = composite(make_context())

    assert result.reward == 0.75


def test_composite_preserves_hard_violation() -> None:
    composite = CompositeRewardTerm(
        terms=(
            lambda context: TowerRewardResult(reward=0.0, hard_violation=False),
            lambda context: TowerRewardResult(reward=0.0, hard_violation=True),
        )
    )

    result = composite(make_context())

    assert result.hard_violation is True


def test_composite_preserves_terminal_success() -> None:
    composite = CompositeRewardTerm(
        terms=(
            lambda context: TowerRewardResult(reward=0.0, is_terminal_success=False),
            lambda context: TowerRewardResult(reward=0.0, is_terminal_success=True),
        )
    )

    result = composite(make_context())

    assert result.is_terminal_success is True


def test_composite_records_child_diagnostics_in_order() -> None:
    composite = CompositeRewardTerm(
        terms=(
            lambda context: TowerRewardResult(
                reward=1.0,
                diagnostics={"name": "first"},
            ),
            lambda context: TowerRewardResult(
                reward=2.0,
                diagnostics={"name": "second"},
            ),
        ),
        diagnostics={"kind": "composite"},
    )

    result = composite(make_context())

    assert result.diagnostics["kind"] == "composite"
    assert result.diagnostics["terms"] == (
        {
            "index": 0,
            "reward": 1.0,
            "hard_violation": False,
            "is_terminal_success": False,
            "diagnostics": {"name": "first"},
        },
        {
            "index": 1,
            "reward": 2.0,
            "hard_violation": False,
            "is_terminal_success": False,
            "diagnostics": {"name": "second"},
        },
    )


def test_composite_rejects_non_reward_result() -> None:
    composite = CompositeRewardTerm(
        terms=(
            lambda context: 1.0,  # type: ignore[return-value]
        )
    )

    with pytest.raises(TypeError, match="reward terms must return TowerRewardResult"):
        composite(make_context())


def test_composite_requires_tuple_terms() -> None:
    with pytest.raises(TypeError, match="terms must be a tuple"):
        CompositeRewardTerm(terms=[])  # type: ignore[arg-type]


def test_success_reward_term_returns_terminal_success_reward() -> None:
    term = SuccessRewardTerm(
        predicate=lambda context: SuccessResult(
            success=True,
            diagnostics={"reason": "success"},
        ),
        success_reward=10.0,
    )

    result = term(make_context())

    assert result.reward == 10.0
    assert result.is_terminal_success is True
    assert result.hard_violation is False
    assert result.diagnostics["success"] == {"reason": "success"}


def test_success_reward_term_returns_failure_reward() -> None:
    term = SuccessRewardTerm(
        predicate=lambda context: SuccessResult(
            success=False,
            diagnostics={"reason": "wrong_root_motion"},
        ),
        success_reward=10.0,
        failure_reward=-1.0,
    )

    result = term(make_context())

    assert result.reward == -1.0
    assert result.is_terminal_success is False
    assert result.diagnostics["success"] == {"reason": "wrong_root_motion"}


def test_success_reward_term_preserves_custom_diagnostics_key_and_metadata() -> None:
    term = SuccessRewardTerm(
        predicate=lambda context: SuccessResult(
            success=True,
            diagnostics={"reason": "success"},
        ),
        success_reward=5.0,
        diagnostics_key="cadence",
        diagnostics={"kind": "terminal"},
    )

    result = term(make_context())

    assert result.diagnostics == {
        "kind": "terminal",
        "cadence": {"reason": "success"},
    }


def test_success_reward_term_rejects_non_success_result() -> None:
    term = SuccessRewardTerm(
        predicate=lambda context: TowerRewardResult(reward=1.0),  # type: ignore[arg-type]
        success_reward=10.0,
    )

    with pytest.raises(TypeError, match="success predicate must return SuccessResult"):
        term(make_context())


def test_composite_can_combine_success_reward_term_with_ordinary_terms() -> None:
    composite = CompositeRewardTerm(
        terms=(
            lambda context: TowerRewardResult(reward=1.0, diagnostics={"kind": "shape"}),
            SuccessRewardTerm(
                predicate=lambda context: SuccessResult(
                    success=True,
                    diagnostics={"reason": "success"},
                ),
                success_reward=10.0,
            ),
        )
    )

    result = composite(make_context())

    assert result.reward == 11.0
    assert result.is_terminal_success is True
    assert result.diagnostics["terms"][1]["is_terminal_success"] is True
