"""Smoke check for the first CounterpointEnv contract."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.envs.counterpoint_env import CounterpointEnv
from rl_counterpoint.graph.actions import StepDelta
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.reward.black_box import ConstantReward


def first_legal_step_delta(
    action_space: tuple[StepDelta, ...],
    action_mask: tuple[bool, ...],
) -> StepDelta:
    for step_delta, is_legal in zip(action_space, action_mask, strict=True):
        if is_legal:
            return step_delta

    raise RuntimeError("no legal StepDelta found")


def print_info_summary(prefix: str, info: dict[str, object]) -> None:
    action_mask = info["action_mask"]
    if not isinstance(action_mask, tuple):
        raise TypeError("action_mask must be a tuple")

    print(f"{prefix} state: {info['state']}")
    print(f"{prefix} step_index: {info['step_index']}")
    print(f"{prefix} target_root_octave: {info['target_root_octave']}")
    print(f"{prefix} action_count: {len(action_mask)}")
    print(f"{prefix} legal_action_count: {sum(action_mask)}")
    print(f"{prefix} has_legal_actions: {info['has_legal_actions']}")


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

    obs, info = env.reset(seed=0)
    print(f"reset obs: {obs}")
    print_info_summary("reset", info)

    action_space = info["action_space"]
    action_mask = info["action_mask"]
    if not isinstance(action_space, tuple) or not isinstance(action_mask, tuple):
        raise TypeError("action_space and action_mask must be tuples")

    step_delta = first_legal_step_delta(action_space, action_mask)
    print(f"chosen StepDelta: {step_delta}")

    obs, reward, terminated, truncated, info = env.step(step_delta)
    print(f"step obs: {obs}")
    print(f"step reward: {reward}")
    print(f"step terminated: {terminated}")
    print(f"step truncated: {truncated}")
    print(f"step valid_action: {info['valid_action']}")
    print(f"step target: {info['target']}")
    print_info_summary("step", info)


if __name__ == "__main__":
    main()
