"""Tests for the first rollout collector."""

from __future__ import annotations

from random import Random

import torch

from rl_counterpoint.algos.rollout import (
    PolicyStepRecord,
    StepRecord,
    choose_masked_logit_action,
    choose_masked_random_action,
    collect_episode,
    collect_policy_episode,
)
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.state_space import is_valid_node
from rl_counterpoint.models.policy import SymbolicChordEncoder, TransformerStepDeltaPolicy
from rl_counterpoint.reward.black_box import ConstantReward


def make_env(*, max_steps: int = 3) -> CounterpointEnv:
    return CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=ConstantReward(reward=1.0),
        initial_state=(3, 6),
        max_steps=max_steps,
        measure_size=4,
        max_step_size=2,
        invalid_action_penalty=-1.0,
    )


class DummyTextEmbedder:
    """Deterministic text embedder for policy-rollout tests."""

    def embed_text(self, text: str) -> torch.Tensor:
        base = float(len(text))
        return torch.tensor([base, base + 1.0, base + 2.0, base + 3.0], dtype=torch.float32)


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
    assert is_valid_node(trajectory[0].observation, env.graph_spec)
    assert trajectory[0].info["target_root_octave"] in {2, 3, 4, 5, 6}
    assert isinstance(trajectory[0].action_index, int)
    assert isinstance(trajectory[0].step_delta, tuple)
    assert isinstance(trajectory[0].action_mask, tuple)
    assert isinstance(trajectory[0].info, dict)


def test_collect_episode_records_next_observation_chain() -> None:
    """Each step record links the environment observation before and after acting."""
    env = make_env(max_steps=2)

    trajectory = collect_episode(env, seed=0)

    assert trajectory[0].next_observation == trajectory[1].observation


def test_choose_masked_logit_action_samples_only_legal_actions() -> None:
    """Masked logit sampling ignores illegal actions entirely."""
    action_space = ((-1, 0), (0, 1), (1, 1))
    action_mask = (False, True, True)
    logits = torch.tensor([100.0, -1.0, 1.0])

    action_index, step_delta = choose_masked_logit_action(
        action_space,
        action_mask,
        logits,
        rng=Random(0),
    )

    assert action_index in (1, 2)
    assert step_delta == action_space[action_index]
    assert action_mask[action_index]


def test_collect_policy_episode_returns_policy_step_records() -> None:
    """Policy-driven rollout records timed-window and logits data per step."""
    env = make_env(max_steps=2)
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())
    policy = TransformerStepDeltaPolicy(
        embedding_dim=4,
        action_dim=len(env.action_space),
        max_window_len=env.measure_size * 3,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )

    trajectory = collect_policy_episode(
        env,
        policy=policy,
        encoder=encoder,
        context_measures=3,
        seed=0,
    )

    assert trajectory
    assert all(isinstance(step, PolicyStepRecord) for step in trajectory)
    assert len(trajectory) == 2
    assert is_valid_node(trajectory[0].observation, env.graph_spec)
    assert trajectory[0].info["target_root_octave"] in {2, 3, 4, 5, 6}
    assert trajectory[0].timed_window.valid_mask[-1]
    assert trajectory[0].logits.shape == (len(env.action_space),)
    assert trajectory[-1].truncated or trajectory[-1].terminated


def test_collect_policy_episode_records_next_observation_chain() -> None:
    """Policy-driven rollout also preserves the observation chain across steps."""
    env = make_env(max_steps=2)
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())
    policy = TransformerStepDeltaPolicy(
        embedding_dim=4,
        action_dim=len(env.action_space),
        max_window_len=env.measure_size * 3,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )

    trajectory = collect_policy_episode(
        env,
        policy=policy,
        encoder=encoder,
        context_measures=3,
        seed=0,
    )

    assert trajectory[0].next_observation == trajectory[1].observation
