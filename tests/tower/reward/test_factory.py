"""Tests for tower reward factory helpers."""

from __future__ import annotations

import pytest

from tower.reward.context import NewFacts, TowerRewardContext
from tower.reward.factory import (
    Rank1RewardFactoryConfig,
    Rank1RewardFunction,
    Rank2RewardFactoryConfig,
    Rank2RewardFunction,
    build_rank1_reward_fn,
    build_rank2_reward_fn,
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
    target_root_octave: int | None = None,
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
        target_root_octave=target_root_octave,
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

    result = reward_fn(make_context(history=((60,), (67,)), action=(-7,)))

    assert result.reward == 11.0
    assert result.is_terminal_success is True
    child_results = result.diagnostics["terms"]
    assert child_results[0]["reward"] == 10.0
    assert child_results[1]["reward"] == 0.0
    assert child_results[2]["reward"] == -0.5
    assert child_results[3]["reward"] == 1.0
    assert child_results[4]["reward"] == 0.5
    assert child_results[5]["reward"] == 0.0


def test_rank1_reward_factory_injects_configured_key() -> None:
    reward_fn = build_rank1_reward_fn(key_pitch_class=2)

    result = reward_fn(
        make_context(
            history=((62,), (69,)),
            action=(-7,),
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
            history=((60,), (67,)),
            action=(-7,),
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

    result = reward_fn(make_context(history=((63,), (70,)), action=(-7,)))

    assert result.diagnostics["kind"] == "rank1_reward"
    assert result.diagnostics["key_pitch_class"] == 3
    assert result.diagnostics["target_root_octave"] == 4
    assert len(result.diagnostics["terms"]) == 6


def test_rank1_reward_factory_injects_configured_target_octave() -> None:
    reward_fn = build_rank1_reward_fn(target_root_octave=5)

    result = reward_fn(make_context(history=((60,),), action=(12,)))

    target_octave = result.diagnostics["terms"][3]["diagnostics"][
        "target_octave_distance"
    ]
    assert result.reward == 1.5
    assert target_octave["root_octave"] == 5
    assert target_octave["target_root_octave"] == 5
    assert target_octave["octave_distance"] == 0


def test_rank1_reward_factory_can_preserve_context_target_octave() -> None:
    reward_fn = build_rank1_reward_fn(
        target_root_octave=5,
        use_context_target_root_octave=True,
    )

    result = reward_fn(
        make_context(
            history=((60,),),
            action=(12,),
            target_root_octave=4,
        )
    )

    target_octave = result.diagnostics["terms"][3]["diagnostics"][
        "target_octave_distance"
    ]
    assert target_octave["root_octave"] == 5
    assert target_octave["target_root_octave"] == 4
    assert target_octave["octave_distance"] == 1


def test_rank1_reward_factory_adds_beat_class_pitch_term() -> None:
    reward_fn = build_rank1_reward_fn(
        measure_start_tonic_reward=1.25,
        onbeat_scale_degree_reward=0.75,
        offbeat_consonance_weight=2.0,
    )

    result = reward_fn(
        make_context(
            history=((59,),),
            action=(1,),
            step_index=0,
            key_pitch_class=None,
            is_final_step=False,
        )
    )

    beat_class_result = result.diagnostics["terms"][4]
    diagnostics = beat_class_result["diagnostics"]["beat_class_pitch"]
    assert beat_class_result["reward"] == 2.0
    assert diagnostics["is_measure_start"] is True
    assert diagnostics["is_tonic"] is True
    assert diagnostics["measure_start_tonic_reward"] == 1.25
    assert diagnostics["onbeat_scale_degree_reward"] == 0.75


def test_rank1_reward_factory_adds_step_size_bin_balance_term() -> None:
    reward_fn = build_rank1_reward_fn(
        step_size_balance_threshold=3,
        step_size_balance_target_small_rate=0.5,
        step_size_balance_weight=2.0,
    )

    result = reward_fn(
        make_context(
            history=((60,), (62,)),
            action=(5,),
            step_index=1,
            is_final_step=False,
        )
    )

    step_balance_result = result.diagnostics["terms"][5]
    diagnostics = step_balance_result["diagnostics"]["step_size_bin_balance"]
    assert step_balance_result["reward"] == 2.0
    assert diagnostics["small_step_threshold"] == 3
    assert diagnostics["target_small_rate"] == 0.5
    assert diagnostics["small_count"] == 1
    assert diagnostics["large_count"] == 1
    assert diagnostics["balance_score"] == 1.0


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

    with pytest.raises(ValueError, match="target_root_octave must be in"):
        Rank1RewardFactoryConfig(target_root_octave=10)

    with pytest.raises(TypeError, match="use_context_target_root_octave must be"):
        Rank1RewardFactoryConfig(use_context_target_root_octave=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_recent_range must be non-negative"):
        build_rank1_reward_fn(max_recent_range=-1)

    with pytest.raises(ValueError, match="large_leap_threshold must be at least 1"):
        build_rank1_reward_fn(large_leap_threshold=0)

    with pytest.raises(ValueError, match="step_size_balance_threshold must be at least 1"):
        Rank1RewardFactoryConfig(step_size_balance_threshold=0)

    with pytest.raises(
        ValueError,
        match="step_size_balance_target_small_rate must be between 0 and 1",
    ):
        Rank1RewardFactoryConfig(step_size_balance_target_small_rate=0.0)


def test_build_rank2_reward_fn_returns_reward_result() -> None:
    reward_fn = build_rank2_reward_fn()

    result = reward_fn(
        make_context(
            history=((67, 71), (60, 64)),
            action=(-7, -7),
            rank=2,
        )
    )

    assert isinstance(reward_fn, Rank2RewardFunction)
    assert isinstance(result, TowerRewardResult)


def test_rank2_reward_factory_combines_goal_vertical_and_cadence_terms() -> None:
    reward_fn = build_rank2_reward_fn(
        terminal_cadence_reward=10.0,
        vertical_consonance_weight=2.0,
        spacing_reward=0.1,
    )

    result = reward_fn(
        TowerRewardContext(
            rank=2,
            step_index=3,
            source=(67, 71),
            target=(60, 64),
            action=(-7, -7),
            window=build_window(
                history=((65, 69), (66, 70)),
                step_index=3,
                measure_size=4,
                context_measures=2,
            ),
            measure_size=4,
            key_pitch_class=0,
            target_root_octave=4,
            is_final_step=True,
            new_facts=NewFacts(new_voice_index=1),
        )
    )

    assert result.is_terminal_success is True
    assert result.reward == pytest.approx(
        10.0 + 1.0 + 2.0 * (1.0 / 9.0) + 0.1
    )
    child_results = result.diagnostics["terms"]
    assert child_results[0]["reward"] == 10.0
    assert child_results[1]["reward"] == 1.0
    assert child_results[2]["reward"] == pytest.approx(2.0 / 9.0)
    assert child_results[3]["reward"] == 0.1


def test_rank2_reward_factory_exposes_top_level_diagnostics() -> None:
    reward_fn = build_rank2_reward_fn(key_pitch_class=3)

    result = reward_fn(
        make_context(
            history=((70, 74), (63, 67)),
            action=(-7, -7),
            rank=2,
        )
    )

    assert result.diagnostics["kind"] == "rank2_reward"
    assert result.diagnostics["key_pitch_class"] == 3
    assert result.diagnostics["target_root_octave"] == 4
    assert len(result.diagnostics["terms"]) == 4


def test_rank2_reward_factory_can_preserve_context_target_octave() -> None:
    reward_fn = build_rank2_reward_fn(
        target_root_octave=5,
        use_context_target_root_octave=True,
    )

    result = reward_fn(
        TowerRewardContext(
            rank=2,
            step_index=0,
            source=(60, 64),
            target=(60, 76),
            action=(0, 12),
            window=build_window(
                history=((60, 64),),
                step_index=0,
                measure_size=4,
                context_measures=2,
            ),
            measure_size=4,
            target_root_octave=4,
            new_facts=NewFacts(new_voice_index=1),
        )
    )

    target_octave = result.diagnostics["terms"][1]["diagnostics"][
        "rank2_target_octave_distance"
    ]
    assert target_octave["root_octave"] == 5
    assert target_octave["target_root_octave"] == 4
    assert target_octave["octave_distance"] == 1


def test_rank2_reward_factory_rejects_non_rank_2_context() -> None:
    reward_fn = build_rank2_reward_fn()

    with pytest.raises(ValueError, match="requires rank 2 context"):
        reward_fn(make_context())


def test_rank2_reward_factory_validates_config_values() -> None:
    with pytest.raises(ValueError, match="key_pitch_class must be in"):
        Rank2RewardFactoryConfig(key_pitch_class=12)
    with pytest.raises(TypeError, match="terminal_cadence_reward must be"):
        Rank2RewardFactoryConfig(terminal_cadence_reward=True)
    with pytest.raises(ValueError, match="target_root_octave must be in"):
        Rank2RewardFactoryConfig(target_root_octave=10)
    with pytest.raises(TypeError, match="use_context_target_root_octave must be"):
        Rank2RewardFactoryConfig(use_context_target_root_octave=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="vertical_consonance_weight must be"):
        Rank2RewardFactoryConfig(vertical_consonance_weight=True)
    with pytest.raises(ValueError, match="upper_register_soft_ceiling must be in"):
        Rank2RewardFactoryConfig(upper_register_soft_ceiling=128)
    with pytest.raises(ValueError, match="min_vertical_gap must be at least 1"):
        Rank2RewardFactoryConfig(min_vertical_gap=0)
