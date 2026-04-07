"""Action helpers for traversing the counterpoint graph G(n)."""

from __future__ import annotations

from typing import TypeAlias

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.non_crossing import is_valid_edge
from rl_counterpoint.graph.state_space import ChordState, iter_node_states

DirectNextStateAction: TypeAlias = ChordState


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
