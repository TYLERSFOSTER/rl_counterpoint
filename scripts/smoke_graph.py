"""Smoke check for direct graph and action inspection."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_counterpoint.graph.actions import (
    StepDelta,
    step_delta_action_mask,
    step_delta_action_space,
    step_delta_to_next_state,
)
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.non_crossing import is_valid_edge
from rl_counterpoint.graph.state_space import (
    ChordState,
    adjacent_intervals,
    is_valid_node,
    outer_interval,
    pitch_class,
)


def first_legal_transition(
    state: ChordState,
    action_space: tuple[StepDelta, ...],
    action_mask: tuple[bool, ...],
) -> tuple[StepDelta, ChordState]:
    """Return the first legal StepDelta and decoded target by mask order."""
    for step_delta, is_legal in zip(action_space, action_mask, strict=True):
        if is_legal:
            return step_delta, step_delta_to_next_state(state, step_delta)

    raise RuntimeError("no legal StepDelta found")


def print_state_summary(
    *,
    state: ChordState,
    spec: CounterpointGraphSpec,
    action_space: tuple[StepDelta, ...],
    action_mask: tuple[bool, ...],
) -> None:
    """Print a compact human-readable summary of one graph node state."""
    if len(action_space) != len(action_mask):
        raise ValueError("action_space and action_mask must have the same length")

    print(f"state: {state}")
    print(f"root_pitch_class: {pitch_class(state[0])}")
    print(f"adjacent_intervals: {adjacent_intervals(state)}")
    print(f"outer_interval: {outer_interval(state)}")
    print(f"is_valid_node: {is_valid_node(state, spec)}")
    print(f"action_count: {len(action_space)}")
    print(f"legal_action_count: {sum(action_mask)}")


def main() -> None:
    spec = CounterpointGraphSpec(n=2, tonic=60)
    state = (3, 6)
    action_space = step_delta_action_space(n=spec.n, max_step_size=2)
    action_mask = step_delta_action_mask(state, action_space, spec)

    print_state_summary(
        state=state,
        spec=spec,
        action_space=action_space,
        action_mask=action_mask,
    )

    step_delta, target = first_legal_transition(state, action_space, action_mask)
    print(f"chosen StepDelta: {step_delta}")
    print(f"candidate target: {target}")
    print(f"is_valid_edge: {is_valid_edge(state, target, spec)}")


if __name__ == "__main__":
    main()
