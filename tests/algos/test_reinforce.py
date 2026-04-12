"""Tests for the first explicit REINFORCE helpers."""

from __future__ import annotations

import torch

from rl_counterpoint.algos.reinforce import (
    ReinforceEpisodeStats,
    discounted_returns,
    masked_log_probability,
    reinforce_loss,
    run_reinforce_episode,
)
from rl_counterpoint.algos.rollout import PolicyStepRecord
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.envs.observation import TimedChordWindow
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.models.policy import SymbolicChordEncoder, TransformerStepDeltaPolicy
from rl_counterpoint.reward.black_box import TargetRootOctaveReward


class DummyTextEmbedder:
    """Deterministic text embedder for REINFORCE tests."""

    def embed_text(self, text: str) -> torch.Tensor:
        base = float(len(text))
        return torch.tensor([base, base + 1.0, base + 2.0, base + 3.0], dtype=torch.float32)


def make_env(*, max_steps: int = 3) -> CounterpointEnv:
    return CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=TargetRootOctaveReward(
            distance_weight=1.0,
            terminal_window_reward=10.0,
        ),
        initial_state=(3, 6),
        max_steps=max_steps,
        measure_size=4,
        max_step_size=2,
        invalid_action_penalty=-1.0,
    )


def make_policy(*, action_dim: int, max_window_len: int) -> TransformerStepDeltaPolicy:
    return TransformerStepDeltaPolicy(
        embedding_dim=4,
        action_dim=action_dim,
        max_window_len=max_window_len,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )


def test_discounted_returns_matches_expected_future_sums() -> None:
    """Discounted returns align each reward with its future discounted sum."""
    returns = discounted_returns((1.0, 2.0, 3.0), gamma=0.5)

    assert torch.allclose(
        returns,
        torch.tensor([2.75, 3.5, 3.0], dtype=torch.float32),
    )


def test_masked_log_probability_uses_only_legal_logits() -> None:
    """The chosen action log-probability is computed inside the legal subset only."""
    logits = torch.tensor([100.0, 1.0, 3.0], dtype=torch.float32)
    action_mask = (False, True, True)

    log_prob = masked_log_probability(
        logits=logits,
        action_mask=action_mask,
        action_index=2,
    )

    expected = torch.log_softmax(torch.tensor([1.0, 3.0]), dim=0)[1]
    assert torch.allclose(log_prob, expected)


def test_reinforce_loss_is_finite_for_policy_trajectory() -> None:
    """One collected trajectory can be replayed into a finite REINFORCE loss."""
    env = make_env(max_steps=2)
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())
    policy = make_policy(action_dim=len(env.action_space), max_window_len=env.measure_size * 3)
    timed_window = TimedChordWindow(
        chord_sequence=((0, 0), (3, 6)),
        bar_positions=(-1, 0),
        valid_mask=(False, True),
    )
    trajectory = [
        PolicyStepRecord(
            observation=(3, 6),
            timed_window=timed_window,
            action_index=0,
            step_delta=env.action_space[0],
            action_mask=tuple(index == 0 for index in range(len(env.action_space))),
            logits=torch.zeros(len(env.action_space)),
            reward=1.0,
            terminated=False,
            truncated=True,
            info={},
            next_observation=(3, 6),
        )
    ]

    loss = reinforce_loss(
        trajectory,
        policy=policy,
        encoder=encoder,
        tonic=env.graph_spec.tonic,
        measure_size=env.measure_size,
        gamma=0.99,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_run_reinforce_episode_returns_stats_and_updates_policy() -> None:
    """One explicit REINFORCE update returns diagnostics and runs end to end."""
    env = make_env(max_steps=2)
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())
    policy = make_policy(action_dim=len(env.action_space), max_window_len=env.measure_size * 3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    stats = run_reinforce_episode(
        env,
        policy=policy,
        encoder=encoder,
        optimizer=optimizer,
        gamma=0.99,
        context_measures=3,
        seed=0,
    )

    assert isinstance(stats, ReinforceEpisodeStats)
    assert stats.episode_length == 2
    assert stats.episode_return > 0.0
    assert stats.mean_step_reward > 0.0
    assert not stats.terminated
    assert stats.truncated
    assert stats.target_root_octave in {2, 3, 4, 5, 6}
    assert isinstance(stats.final_root_octave, int)
    assert stats.final_octave_distance == abs(
        stats.final_root_octave - stats.target_root_octave
    )
    assert stats.hit_target_on_final_step == (stats.final_octave_distance == 0)
    assert torch.isfinite(torch.tensor(stats.loss))
