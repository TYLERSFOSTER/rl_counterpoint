"""Tests for the first rollout collector."""

from __future__ import annotations

from random import Random

from rl_counterpoint.algos.rollout import (
    StepRecord,
    choose_masked_random_action,
    collect_episode,
)
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.reward.black_box import ConstantReward


def make_env(*, max_steps: int = 3) -> CounterpointEnv:
    return CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=ConstantReward(reward=1.0),
        initial_state=(3, 6),
        max_steps=max_steps,
        max_step_size=2,
        invalid_action_penalty=-1.0,
    )


def test_choose_masked_random_action_returns_legal_action() -> None:
    """Random action selection chooses only among legal masked actions."""
    action_space = ((-1, 0), (0, 1), (1, 1))
    action_mask = (False, True, True)

    action_index, step_delta = choose_masked_random_action(
        action_space,
        action_mask,
        rng=Random(0),
    )

    assert action_index in (1, 2)
    assert step_delta == action_space[action_index]
    assert action_mask[action_index]


def test_collect_episode_returns_explicit_step_records() -> None:
    """One collected episode is a list of explicit environment step records."""
    env = make_env(max_steps=2)

    trajectory = collect_episode(env, seed=0)

    assert trajectory
    assert all(isinstance(step, StepRecord) for step in trajectory)
    assert trajectory[-1].truncated or trajectory[-1].terminated
    assert len(trajectory) == 2
    assert trajectory[0].observation == (3, 6)
    assert isinstance(trajectory[0].action_index, int)
    assert isinstance(trajectory[0].step_delta, tuple)
    assert isinstance(trajectory[0].action_mask, tuple)
    assert isinstance(trajectory[0].info, dict)


def test_collect_episode_records_next_observation_chain() -> None:
    """Each step record links the environment observation before and after acting."""
    env = make_env(max_steps=2)

    trajectory = collect_episode(env, seed=0)

    assert trajectory[0].next_observation == trajectory[1].observation
