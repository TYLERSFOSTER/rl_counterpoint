"""Graph statistics script using the live training graph construction path."""

from __future__ import annotations

from math import ceil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.envs.counterpoint_env import TARGET_ROOT_OCTAVE_CHOICES
from rl_counterpoint.graph.actions import step_delta_action_mask
from rl_counterpoint.graph.state_space import ChordState, iter_node_states
from rl_counterpoint.reward.black_box import midi_to_octave
from scripts.train_reinforce import TrainConfig, build_env


def legal_out_star_size(
    state: ChordState,
    *,
    action_space: tuple[tuple[int, ...], ...],
    graph_spec,
) -> int:
    """Return the number of legal outgoing step-delta actions from one node."""
    return sum(step_delta_action_mask(state, action_space, graph_spec))


def average_out_star_size(config: TrainConfig) -> float:
    """Return the average legal out-degree over the full node set G(n)_0."""
    env = build_env(config)
    states = iter_node_states(env.graph_spec)
    if not states:
        raise ValueError("graph has no valid node states")

    total_out_degree = sum(
        legal_out_star_size(
            state,
            action_space=env.action_space,
            graph_spec=env.graph_spec,
        )
        for state in states
    )
    return total_out_degree / len(states)


def rough_goal_step_count(config: TrainConfig) -> int:
    """Return a coarse root-motion step count using max-size horizontal steps."""
    env = build_env(config)
    initial_state = env.initial_state
    if initial_state is None:
        raise ValueError("graph stats require a concrete initial_state")

    initial_root_octave = midi_to_octave(initial_state[0])
    furthest_target_root_octave = max(
        TARGET_ROOT_OCTAVE_CHOICES,
        key=lambda octave: abs(octave - initial_root_octave),
    )
    semitone_distance = abs(furthest_target_root_octave - initial_root_octave) * 12
    return max(1, ceil(semitone_distance / config.max_step_size))


def branch_growth_estimate(config: TrainConfig) -> float:
    """Return the rough walk-space scale b^d using a heuristic depth d."""
    average_branch = average_out_star_size(config)
    heuristic_depth = 1.5 * rough_goal_step_count(config)
    return average_branch**heuristic_depth


def main() -> None:
    config = TrainConfig()
    average = average_out_star_size(config)
    rough_steps = rough_goal_step_count(config)
    heuristic_depth = 1.5 * rough_steps
    growth = branch_growth_estimate(config)
    print(f"voice_count: {config.voice_count}")
    print(f"max_step_size: {config.max_step_size}")
    print(f"average_out_star_size: {average}")
    print(f"rough_goal_step_count: {rough_steps}")
    print(f"heuristic_depth_d: {heuristic_depth}")
    print(f"branch_growth_estimate_b_to_d: {growth}")


if __name__ == "__main__":
    main()
