"""Tests for the first Gymnasium-style CounterpointEnv contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.state_space import is_valid_node
from rl_counterpoint.reward.protocol import RewardContext, RewardResult
from rl_counterpoint.graph.actions import step_delta_to_next_state
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.reward.black_box import ConstantReward


@dataclass
class RecordingReward:
    """Test helper that records the last reward call context."""

    contexts: list[RewardContext] = field(default_factory=list)

    def __call__(
        self,
        source: tuple[int, ...],
        target: tuple[int, ...],
        context: RewardContext,
    ) -> RewardResult:
        self.contexts.append(context)
        return RewardResult(reward=0.0)


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


def first_legal_step_delta(info: dict[str, object]) -> tuple[int, ...]:
    action_space = info["action_space"]
    action_mask = info["action_mask"]
    assert isinstance(action_space, tuple)
    assert isinstance(action_mask, tuple)

    for step_delta, is_legal in zip(action_space, action_mask, strict=True):
        if is_legal:
            return step_delta

    raise RuntimeError("expected at least one legal StepDelta")


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

    obs, info = env.reset(seed=0)

    assert obs == info["state"]
    assert is_valid_node(obs, env.graph_spec)
    assert info["step_index"] == 0
    assert info["measure_size"] == 4
    assert info["bar_position"] == 0
    assert info["is_leading_beat"]
    assert info["is_downbeat"]
    assert not info["is_ending_beat"]
    assert info["history"] == (obs,)
    assert info["target_root_octave"] in {2, 3, 4, 5, 6}
    assert info["reward_deadline_step"] >= 4
    assert info["reward_deadline_measures"] >= 1
    assert info["action_space"]
    assert info["action_mask"]
    assert len(info["action_space"]) == len(info["action_mask"])
    assert info["has_legal_actions"]


def test_valid_step_updates_state_and_calls_reward() -> None:
    """A valid StepDelta updates the state and returns reward diagnostics."""
    env = make_env()
    initial_obs, initial_info = env.reset(seed=0)
    step_delta = first_legal_step_delta(initial_info)
    expected_target = step_delta_to_next_state(initial_obs, step_delta)

    obs, reward, terminated, truncated, info = env.step(step_delta)

    assert obs == expected_target
    assert env.state == expected_target
    assert reward == 1.25
    assert not terminated
    assert not truncated
    assert info["source"] == initial_obs
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
    assert info["history"] == (initial_obs, expected_target)
    assert info["target_root_octave"] == initial_info["target_root_octave"]


def test_invalid_step_delta_is_noop_with_penalty_and_diagnostic() -> None:
    """An invalid StepDelta is not projected into a legal transition."""
    env = make_env(invalid_action_penalty=-3.5)
    initial_obs, initial_info = env.reset(seed=0)

    obs, reward, terminated, truncated, info = env.step((0, 0))

    assert obs == initial_obs
    assert env.state == initial_obs
    assert reward == -3.5
    assert not terminated
    assert not truncated
    assert not info["valid_action"]
    assert info["source"] == initial_obs
    assert info["target"] == initial_obs
    assert info["step_delta"] == (0, 0)
    assert info["invalid_action_reason"] == "decoded target is not a valid edge"
    assert info["measure_size"] == 4
    assert info["bar_position"] == 1
    assert info["history"] == (initial_obs, initial_obs)
    assert info["target_root_octave"] == initial_info["target_root_octave"]


def test_step_truncates_at_max_steps() -> None:
    """Episodes truncate when the max-step limit is reached."""
    env = make_env(max_steps=1)
    _obs, info = env.reset(seed=0)
    step_delta = first_legal_step_delta(info)

    _obs, _reward, terminated, truncated, info = env.step(step_delta)

    assert not terminated
    assert truncated
    assert info["step_index"] == 1


def test_step_truncates_at_reward_deadline_when_sooner_than_max_steps() -> None:
    """The reward deadline becomes the effective truncation limit when earlier."""
    env = CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=ConstantReward(reward=1.25),
        initial_state=None,
        max_steps=16,
        measure_size=4,
        max_step_size=12,
    )
    _obs, info = env.reset(seed=0)
    deadline_step = info["reward_deadline_step"]
    assert deadline_step <= 16

    for _ in range(deadline_step - 1):
        step_delta = first_legal_step_delta(info)
        _obs, _reward, terminated, truncated, info = env.step(step_delta)
        assert not terminated
        assert not truncated

    step_delta = first_legal_step_delta(info)
    _obs, _reward, terminated, truncated, info = env.step(step_delta)

    assert not terminated
    assert truncated
    assert info["step_index"] == deadline_step
    assert info["reward_deadline_step"] == deadline_step


def test_valid_step_passes_extended_reward_context() -> None:
    """The env passes step delta, key pitch class, and timed window to reward."""
    reward_fn = RecordingReward()
    env = CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=reward_fn,
        initial_state=(3, 6),
        max_steps=4,
        measure_size=4,
        max_step_size=2,
    )
    initial_obs, info = env.reset(seed=0)
    step_delta = first_legal_step_delta(info)

    _obs, _reward, _terminated, _truncated, _info = env.step(step_delta)

    assert len(reward_fn.contexts) == 1
    context = reward_fn.contexts[0]
    assert context.step_delta == step_delta
    assert context.key_pitch_class == 0
    assert context.measure_size == 4
    assert context.max_step_size == 2
    assert context.history == (initial_obs,)
    assert context.target_root_octave == env.target_root_octave
    assert not context.is_final_step
    assert context.timed_chord_window is not None
    assert context.timed_chord_window.chord_sequence[-1] == initial_obs
    assert context.timed_chord_window.bar_positions == (
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
    )
    assert context.timed_chord_window.valid_mask == (
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    )


def test_reset_seed_makes_start_state_and_target_reproducible() -> None:
    """Reset seeding reproduces the sampled start state and target octave."""
    env = make_env()

    first_obs, first_info = env.reset(seed=123)
    second_obs, second_info = env.reset(seed=123)

    assert first_obs == second_obs
    assert first_info["target_root_octave"] == second_info["target_root_octave"]


def test_target_root_octave_persists_across_episode_steps() -> None:
    """One reset target octave remains stable until the next reset."""
    env = make_env()
    _obs, info = env.reset(seed=0)
    target_root_octave = info["target_root_octave"]
    step_delta = first_legal_step_delta(info)

    _obs, _reward, _terminated, _truncated, step_info = env.step(step_delta)

    assert env.target_root_octave == target_root_octave
    assert step_info["target_root_octave"] == target_root_octave
