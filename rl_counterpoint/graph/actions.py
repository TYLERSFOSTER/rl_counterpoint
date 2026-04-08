"""Action helpers for traversing the counterpoint graph G(n)."""

from __future__ import annotations

from itertools import product
from typing import TypeAlias

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.non_crossing import is_valid_edge
from rl_counterpoint.graph.state_space import ChordState, iter_node_states

DirectNextStateAction: TypeAlias = ChordState
StepDelta: TypeAlias = tuple[int, ...]


def action_to_next_state(action: DirectNextStateAction) -> ChordState:
    """Interpret an action as a direct next-state choice.

    This is the simplest possible action representation. It is intentionally
    thin so it can be replaced if the project owner chooses a factorized
    per-voice action representation later.
    """
    return action


def is_valid_action(
    current_state: ChordState,
    action: DirectNextStateAction,
    spec: CounterpointGraphSpec,
) -> bool:
    return is_valid_edge(current_state, action_to_next_state(action), spec)


def candidate_next_states(
    current_state: ChordState,
    spec: CounterpointGraphSpec,
) -> tuple[ChordState, ...]:
    """Return direct next-state candidates reachable from current_state."""
    return tuple(
        candidate
        for candidate in iter_node_states(spec)
        if is_valid_edge(current_state, candidate, spec)
    )


def action_mask(
    current_state: ChordState,
    action_space: tuple[DirectNextStateAction, ...],
    spec: CounterpointGraphSpec,
) -> tuple[bool, ...]:
    """Return a Boolean legality mask for a supplied direct-state action space."""
    return tuple(is_valid_action(current_state, action, spec) for action in action_space)


def step_delta_to_next_state(
    current_state: ChordState,
    step_delta: StepDelta,
) -> ChordState:
    """Decode a step delta as signed per-voice change in state."""
    if len(step_delta) != len(current_state):
        raise ValueError("step_delta length must match current_state length")

    return tuple(
        pitch + delta
        for pitch, delta in zip(current_state, step_delta, strict=True)
    )


def step_delta_action_space(
    *,
    n: int,
    max_step_size: int,
) -> tuple[StepDelta, ...]:
    """Return the nonzero bounded step-delta lattice for n voices."""
    if n < 1:
        raise ValueError("n must be at least 1")

    if max_step_size < 1:
        raise ValueError("max_step_size must be at least 1")

    deltas = range(-max_step_size, max_step_size + 1)
    zero_delta = (0,) * n
    return tuple(
        step_delta
        for step_delta in product(deltas, repeat=n)
        if step_delta != zero_delta
    )


def is_valid_step_delta_action(
    current_state: ChordState,
    step_delta: StepDelta,
    spec: CounterpointGraphSpec,
) -> bool:
    """Return True iff the step delta decodes to a valid graph edge."""
    target = step_delta_to_next_state(current_state, step_delta)
    return is_valid_edge(current_state, target, spec)


def candidate_step_delta_actions(
    current_state: ChordState,
    spec: CounterpointGraphSpec,
    *,
    max_step_size: int,
) -> tuple[StepDelta, ...]:
    """Return bounded step deltas that decode to valid graph edges."""
    return tuple(
        step_delta
        for step_delta in step_delta_action_space(
            n=spec.n,
            max_step_size=max_step_size,
        )
        if is_valid_step_delta_action(current_state, step_delta, spec)
    )


def step_delta_action_mask(
    current_state: ChordState,
    action_space: tuple[StepDelta, ...],
    spec: CounterpointGraphSpec,
) -> tuple[bool, ...]:
    """Return a Boolean legality mask for a supplied step-delta action space."""
    return tuple(
        is_valid_step_delta_action(current_state, step_delta, spec)
        for step_delta in action_space
    )
