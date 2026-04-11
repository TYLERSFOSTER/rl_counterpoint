"""Smoke check for direct reward inspection."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.envs.observation import TimedChordWindow
from rl_counterpoint.graph.state_space import ChordState
from rl_counterpoint.reward.black_box import TargetRootOctaveReward, midi_to_octave
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
    print(f"root_octave: {diagnostics['root_octave']}")
    print(f"target_root_octave: {diagnostics['target_root_octave']}")
    print(f"octave_distance: {diagnostics['octave_distance']}")
    print(f"distance_reward: {diagnostics['distance_reward']}")
    print(f"is_final_step: {diagnostics['is_final_step']}")
    print(f"terminal_bonus: {diagnostics['terminal_bonus']}")
    print(f"terminal_match: {diagnostics['terminal_match']}")

def make_example_context(
    *,
    step_index: int,
    target_root_octave: int,
    is_final_step: bool,
) -> RewardContext:
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
        target_root_octave=target_root_octave,
        is_final_step=is_final_step,
    )


def main() -> None:
    reward_fn = TargetRootOctaveReward(distance_weight=1.0, terminal_match_reward=10.0)
    source = (60, 64, 67)
    near_target = (60, 64, 67)
    exact_final_target = (72, 76, 79)

    shaping_result = reward_fn(
        source,
        near_target,
        make_example_context(
            step_index=1,
            target_root_octave=5,
            is_final_step=False,
        ),
    )
    final_hit_result = reward_fn(
        source,
        exact_final_target,
        make_example_context(
            step_index=7,
            target_root_octave=5,
            is_final_step=True,
        ),
    )

    print("distance-shaping example")
    print(f"target root octave from target chord: {midi_to_octave(near_target[0])}")
    print_reward_summary(source=source, target=near_target, result=shaping_result)
    print("final-hit example")
    print(
        f"target root octave from target chord: {midi_to_octave(exact_final_target[0])}"
    )
    print_reward_summary(source=source, target=exact_final_target, result=final_hit_result)


if __name__ == "__main__":
    main()
