"""Tests for the first Gymnasium-style CounterpointEnv contract."""

from __future__ import annotations

import pytest

from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.actions import step_delta_to_next_state
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.reward.black_box import ConstantReward


def make_env(
    *,
    max_steps: int = 4,
    measure_size: int = 4,
    invalid_action_penalty: float = -2.0,
) -> CounterpointEnv:
    return CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=ConstantReward(reward=1.25),
        initial_state=(3, 6),
        max_steps=max_steps,
        measure_size=measure_size,
        max_step_size=2,
        invalid_action_penalty=invalid_action_penalty,
    )


def test_constructor_rejects_invalid_initial_state() -> None:
    """The constructor-provided initial state must belong to G(n)_0."""
    with pytest.raises(ValueError, match="initial_state must be a valid graph node"):
        CounterpointEnv(
            graph_spec=CounterpointGraphSpec(n=2, tonic=60),
            reward_fn=ConstantReward(),
            initial_state=(0, 3),
            max_steps=4,
            measure_size=4,
            max_step_size=2,
        )


def test_constructor_rejects_invalid_max_steps() -> None:
    """The environment needs a positive max-step truncation limit."""
    with pytest.raises(ValueError, match="max_steps must be at least 1"):
        CounterpointEnv(
            graph_spec=CounterpointGraphSpec(n=2, tonic=60),
            reward_fn=ConstantReward(),
            initial_state=(3, 6),
            max_steps=0,
            measure_size=4,
            max_step_size=2,
        )


def test_constructor_rejects_invalid_measure_size() -> None:
    """The environment needs a positive m in the meter contract m/4."""
    with pytest.raises(ValueError, match="measure_size must be at least 1"):
        CounterpointEnv(
            graph_spec=CounterpointGraphSpec(n=2, tonic=60),
            reward_fn=ConstantReward(),
            initial_state=(3, 6),
            max_steps=4,
            measure_size=0,
            max_step_size=2,
        )


def test_reset_returns_raw_state_and_action_info() -> None:
    """Reset returns the raw ChordState observation plus mask diagnostics."""
    env = make_env()

    obs, info = env.reset()

    assert obs == (3, 6)
    assert info["state"] == (3, 6)
    assert info["step_index"] == 0
    assert info["measure_size"] == 4
    assert info["bar_position"] == 0
    assert info["is_leading_beat"]
    assert info["is_downbeat"]
    assert not info["is_ending_beat"]
    assert info["history"] == ((3, 6),)
    assert info["action_space"]
    assert info["action_mask"]
    assert len(info["action_space"]) == len(info["action_mask"])
    assert info["has_legal_actions"]


def test_valid_step_updates_state_and_calls_reward() -> None:
    """A valid StepDelta updates the state and returns reward diagnostics."""
    env = make_env()
    env.reset()
    step_delta = (1, 1)
    expected_target = step_delta_to_next_state((3, 6), step_delta)

    obs, reward, terminated, truncated, info = env.step(step_delta)

    assert obs == expected_target
    assert env.state == expected_target
    assert reward == 1.25
    assert not terminated
    assert not truncated
    assert info["source"] == (3, 6)
    assert info["target"] == expected_target
    assert info["step_delta"] == step_delta
    assert info["valid_action"]
    assert info["reward_diagnostics"]["kind"] == "constant"
    assert info["step_index"] == 1
    assert info["measure_size"] == 4
    assert info["bar_position"] == 1
    assert not info["is_leading_beat"]
    assert not info["is_downbeat"]
    assert not info["is_ending_beat"]
    assert info["history"] == ((3, 6), expected_target)


def test_invalid_step_delta_is_noop_with_penalty_and_diagnostic() -> None:
    """An invalid StepDelta is not projected into a legal transition."""
    env = make_env(invalid_action_penalty=-3.5)
    env.reset()

    obs, reward, terminated, truncated, info = env.step((0, 0))

    assert obs == (3, 6)
    assert env.state == (3, 6)
    assert reward == -3.5
    assert not terminated
    assert not truncated
    assert not info["valid_action"]
    assert info["source"] == (3, 6)
    assert info["target"] == (3, 6)
    assert info["step_delta"] == (0, 0)
    assert info["invalid_action_reason"] == "decoded target is not a valid edge"
    assert info["measure_size"] == 4
    assert info["bar_position"] == 1
    assert info["history"] == ((3, 6), (3, 6))


def test_step_truncates_at_max_steps() -> None:
    """Episodes truncate when the max-step limit is reached."""
    env = make_env(max_steps=1)
    env.reset()

    _obs, _reward, terminated, truncated, info = env.step((1, 1))

    assert not terminated
    assert truncated
    assert info["step_index"] == 1
