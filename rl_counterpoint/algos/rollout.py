"""Rollout collection for the first counterpoint environment."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import torch

from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.envs.observation import TimedChordWindow, build_timed_chord_window
from rl_counterpoint.graph.actions import StepDelta
from rl_counterpoint.graph.state_space import ChordState
from rl_counterpoint.models.policy import (
    SymbolicChordEncoder,
    TransformerStepDeltaPolicy,
    encode_timed_chord_window,
)


Info = dict[str, Any]


@dataclass(frozen=True)
class StepRecord:
    """One explicit environment transition collected for later learning."""

    observation: ChordState
    action_index: int
    step_delta: StepDelta
    action_mask: tuple[bool, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: Info
    next_observation: ChordState


@dataclass(frozen=True)
class PolicyStepRecord:
    """One explicit policy-driven transition under the sequence observation contract."""

    observation: ChordState
    timed_window: TimedChordWindow
    action_index: int
    step_delta: StepDelta
    action_mask: tuple[bool, ...]
    logits: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    info: Info
    next_observation: ChordState


def choose_masked_random_action(
    action_space: tuple[StepDelta, ...],
    action_mask: tuple[bool, ...],
    *,
    rng: Random,
) -> tuple[int, StepDelta]:
    """Choose uniformly among legal masked actions."""
    legal_indices = [
        index
        for index, is_legal in enumerate(action_mask)
        if is_legal
    ]
    if not legal_indices:
        raise RuntimeError("no legal StepDelta available")

    action_index = rng.choice(legal_indices)
    return action_index, action_space[action_index]


def choose_masked_logit_action(
    action_space: tuple[StepDelta, ...],
    action_mask: tuple[bool, ...],
    logits: torch.Tensor,
    *,
    rng: Random,
) -> tuple[int, StepDelta]:
    """Sample one legal action from policy logits under an external legality mask."""
    if logits.ndim != 1:
        raise ValueError("logits must be rank 1 [action_dim]")
    if logits.shape[0] != len(action_space):
        raise ValueError("logits length must match action_space length")
    if len(action_mask) != len(action_space):
        raise ValueError("action_mask length must match action_space length")

    legal_indices = [
        index
        for index, is_legal in enumerate(action_mask)
        if is_legal
    ]
    if not legal_indices:
        raise RuntimeError("no legal StepDelta available")

    legal_logits = logits[legal_indices]
    if not torch.isfinite(legal_logits).all():
        raise ValueError("legal logits must all be finite")

    probabilities = torch.softmax(legal_logits, dim=0).tolist()
    action_index = rng.choices(legal_indices, weights=probabilities, k=1)[0]
    return action_index, action_space[action_index]


def collect_episode(
    env: CounterpointEnv,
    *,
    seed: int = 0,
) -> list[StepRecord]:
    """Collect one trajectory using a masked random policy."""
    rng = Random(seed)
    observation, info = env.reset(seed=seed)
    trajectory: list[StepRecord] = []

    while True:
        action_space = info["action_space"]
        action_mask = info["action_mask"]

        if not isinstance(action_space, tuple) or not isinstance(action_mask, tuple):
            raise TypeError("action_space and action_mask must be tuples")

        action_index, step_delta = choose_masked_random_action(
            action_space,
            action_mask,
            rng=rng,
        )
        next_observation, reward, terminated, truncated, next_info = env.step(step_delta)
        trajectory.append(
            StepRecord(
                observation=observation,
                action_index=action_index,
                step_delta=step_delta,
                action_mask=action_mask,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=next_info,
                next_observation=next_observation,
            )
        )

        if terminated or truncated:
            return trajectory

        observation = next_observation
        info = next_info


def collect_policy_episode(
    env: CounterpointEnv,
    *,
    policy: TransformerStepDeltaPolicy,
    encoder: SymbolicChordEncoder,
    context_measures: int = 3,
    seed: int | None = None,
) -> list[PolicyStepRecord]:
    """Collect one trajectory using the sequence-policy rollout path."""
    rng = Random(seed)
    observation, info = env.reset(seed=seed)
    trajectory: list[PolicyStepRecord] = []

    while True:
        timed_window = build_timed_chord_window(
            history=env.history,
            step_index=env.step_index,
            measure_size=env.measure_size,
            context_measures=context_measures,
        )
        encoded_window = encode_timed_chord_window(
            window=timed_window,
            tonic=env.graph_spec.tonic,
            measure_size=env.measure_size,
            encoder=encoder,
            target_root_octave=info.get("target_root_octave"),
        )
        logits = policy(encoded_window)
        action_space = info["action_space"]
        action_mask = info["action_mask"]

        if not isinstance(action_space, tuple) or not isinstance(action_mask, tuple):
            raise TypeError("action_space and action_mask must be tuples")

        action_index, step_delta = choose_masked_logit_action(
            action_space,
            action_mask,
            logits,
            rng=rng,
        )
        next_observation, reward, terminated, truncated, next_info = env.step(step_delta)
        trajectory.append(
            PolicyStepRecord(
                observation=observation,
                timed_window=timed_window,
                action_index=action_index,
                step_delta=step_delta,
                action_mask=action_mask,
                logits=logits.detach().clone(),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=next_info,
                next_observation=next_observation,
            )
        )

        if terminated or truncated:
            return trajectory

        observation = next_observation
        info = next_info
