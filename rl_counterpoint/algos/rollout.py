"""Rollout collection for the first counterpoint environment."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.actions import StepDelta
from rl_counterpoint.graph.state_space import ChordState


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


def collect_episode(
    env: CounterpointEnv,
    *,
    seed: int = 0,
) -> list[StepRecord]:
    """Collect one trajectory using a masked random policy."""
    rng = Random(seed)
    observation, info = env.reset()
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
