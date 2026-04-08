"""Smoke check for one masked-random rollout episode."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.algos.rollout import StepRecord, collect_episode
from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.reward.black_box import ConstantReward


def print_step_record(index: int, step: StepRecord) -> None:
    """Print a compact human-readable summary of one rollout step."""
    legal_action_count = sum(step.action_mask)
    print(f"step {index} observation: {step.observation}")
    print(f"step {index} action_index: {step.action_index}")
    print(f"step {index} step_delta: {step.step_delta}")
    print(f"step {index} legal_action_count: {legal_action_count}")
    print(f"step {index} reward: {step.reward}")
    print(f"step {index} next_observation: {step.next_observation}")
    print(f"step {index} terminated: {step.terminated}")
    print(f"step {index} truncated: {step.truncated}")


def main() -> None:
    env = CounterpointEnv(
        graph_spec=CounterpointGraphSpec(n=2, tonic=60),
        reward_fn=ConstantReward(reward=1.0),
        initial_state=(3, 6),
        max_steps=4,
        measure_size=4,
        max_step_size=2,
        invalid_action_penalty=-1.0,
    )

    trajectory = collect_episode(env, seed=0)
    print(f"episode length: {len(trajectory)}")
    for index, step in enumerate(trajectory):
        print_step_record(index, step)


if __name__ == "__main__":
    main()
