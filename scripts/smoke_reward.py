"""Smoke check for direct reward inspection."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.envs.observation import TimedChordWindow
from rl_counterpoint.graph.state_space import ChordState
from rl_counterpoint.reward.black_box import StrongBeatConsonanceReward
from rl_counterpoint.reward.protocol import RewardContext, RewardResult


def print_reward_summary(
    *,
    source: ChordState,
    target: ChordState,
    result: RewardResult,
) -> None:
    """Print a compact human-readable summary of one reward evaluation."""
    diagnostics = result.diagnostics
    print(f"source: {source}")
    print(f"target: {target}")
    print(f"reward: {result.reward}")
    print(f"kind: {diagnostics['kind']}")
    print(f"step_index: {diagnostics['step_index']}")
    print(f"is_strong_beat: {diagnostics['is_strong_beat']}")
    print(f"applied_beat_weight: {diagnostics['applied_beat_weight']}")
    print(f"base_static_consonance_reward: {diagnostics['base_static_consonance_reward']}")


def make_example_context(*, step_index: int) -> RewardContext:
    """Build one deterministic reward context for smoke inspection."""
    return RewardContext(
        step_index=step_index,
        max_steps=8,
        measure_size=4,
        history=((60, 64, 67),),
        step_delta=(0, 1, 0),
        key_pitch_class=0,
        timed_chord_window=TimedChordWindow(
            chord_sequence=((0, 0, 0), (60, 64, 67)),
            bar_positions=(-1, step_index % 4),
            valid_mask=(False, True),
        ),
    )


def main() -> None:
    reward_fn = StrongBeatConsonanceReward()
    source = (60, 64, 67)
    target = (60, 64, 67)

    strong_result = reward_fn(source, target, make_example_context(step_index=0))
    weak_result = reward_fn(source, target, make_example_context(step_index=1))

    print("strong-beat example")
    print_reward_summary(source=source, target=target, result=strong_result)
    print("weak-beat example")
    print_reward_summary(source=source, target=target, result=weak_result)


if __name__ == "__main__":
    main()
