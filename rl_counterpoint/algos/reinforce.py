"""Tiny explicit REINFORCE helpers for the sequence-policy path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_counterpoint.algos.rollout import PolicyStepRecord, collect_policy_episode
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.models.policy import (
    SymbolicChordEncoder,
    TransformerStepDeltaPolicy,
    encode_timed_chord_window,
)
from rl_counterpoint.reward.black_box import midi_to_octave


@dataclass(frozen=True)
class ReinforceEpisodeStats:
    """Compact diagnostics for one explicit REINFORCE update."""

    episode_return: float
    episode_length: int
    mean_step_reward: float
    terminated: bool
    truncated: bool
    loss: float
    target_root_octave: int
    final_root_octave: int
    final_octave_distance: int
    hit_target_on_final_step: bool


def discounted_returns(
    rewards: tuple[float, ...],
    *,
    gamma: float,
) -> Tensor:
    """Return discounted returns aligned with one episode's reward sequence."""
    if not rewards:
        raise ValueError("rewards must not be empty")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0.0, 1.0]")

    running_return = 0.0
    returns = []
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def masked_log_probability(
    *,
    logits: Tensor,
    action_mask: tuple[bool, ...],
    action_index: int,
) -> Tensor:
    """Return log pi(a|x) under an external legality mask."""
    if logits.ndim != 1:
        raise ValueError("logits must be rank 1 [action_dim]")
    if len(action_mask) != logits.shape[0]:
        raise ValueError("action_mask length must match logits length")
    if action_index < 0 or action_index >= logits.shape[0]:
        raise ValueError("action_index is out of range")
    if not action_mask[action_index]:
        raise ValueError("chosen action_index must be legal under action_mask")

    legal_indices = [
        index
        for index, is_legal in enumerate(action_mask)
        if is_legal
    ]
    if not legal_indices:
        raise RuntimeError("no legal StepDelta available")

    legal_logits = logits[legal_indices]
    legal_log_probs = torch.log_softmax(legal_logits, dim=0)
    legal_position = legal_indices.index(action_index)
    return legal_log_probs[legal_position]


def reinforce_loss(
    trajectory: list[PolicyStepRecord],
    *,
    policy: TransformerStepDeltaPolicy,
    encoder: SymbolicChordEncoder,
    tonic: int,
    measure_size: int,
    gamma: float,
) -> Tensor:
    """Compute the REINFORCE loss by replaying one collected trajectory."""
    if not trajectory:
        raise ValueError("trajectory must not be empty")

    returns = discounted_returns(
        tuple(step.reward for step in trajectory),
        gamma=gamma,
    )
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

    losses = []
    for step, return_t in zip(trajectory, returns, strict=True):
        encoded_window = encode_timed_chord_window(
            window=step.timed_window,
            tonic=tonic,
            measure_size=measure_size,
            encoder=encoder,
            target_root_octave=step.info.get("target_root_octave"),
        )
        logits = policy(encoded_window)
        log_prob = masked_log_probability(
            logits=logits,
            action_mask=step.action_mask,
            action_index=step.action_index,
        )
        losses.append(-log_prob * return_t)

    return torch.stack(losses).sum()


def run_reinforce_episode(
    env: CounterpointEnv,
    *,
    policy: TransformerStepDeltaPolicy,
    encoder: SymbolicChordEncoder,
    optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    context_measures: int = 3,
    seed: int = 0,
) -> ReinforceEpisodeStats:
    """Collect one episode and perform one explicit REINFORCE update."""
    trajectory = collect_policy_episode(
        env,
        policy=policy,
        encoder=encoder,
        context_measures=context_measures,
        seed=seed,
    )
    loss = reinforce_loss(
        trajectory,
        policy=policy,
        encoder=encoder,
        tonic=env.graph_spec.tonic,
        measure_size=env.measure_size,
        gamma=gamma,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    final_step = trajectory[-1]
    target_root_octave = int(final_step.info["target_root_octave"])
    final_root_octave = midi_to_octave(final_step.next_observation[0])
    final_octave_distance = abs(final_root_octave - target_root_octave)

    return ReinforceEpisodeStats(
        episode_return=sum(step.reward for step in trajectory),
        episode_length=len(trajectory),
        mean_step_reward=sum(step.reward for step in trajectory) / len(trajectory),
        terminated=trajectory[-1].terminated,
        truncated=trajectory[-1].truncated,
        loss=float(loss.item()),
        target_root_octave=target_root_octave,
        final_root_octave=final_root_octave,
        final_octave_distance=final_octave_distance,
        hit_target_on_final_step=final_octave_distance == 0,
    )
