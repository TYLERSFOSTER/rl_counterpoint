"""Tests for graph action helpers.

These tests cover the current minimal action representation: a direct
next-state choice. They verify that action decoding is transparent, action
validity agrees with the graph edge predicate, generated candidates are legal,
and legality masks align with a supplied direct-state action space.
"""

from __future__ import annotations

from rl_counterpoint.graph.actions import (
    action_mask,
    action_to_next_state,
    candidate_next_states,
    is_valid_action,
)
from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.non_crossing import is_valid_edge
from rl_counterpoint.graph.state_space import iter_node_states


def test_action_to_next_state_returns_direct_state_action() -> None:
    """The current direct action representation decodes to itself."""
    action = (3, 6)

    assert action_to_next_state(action) == action


def test_is_valid_action_agrees_with_is_valid_edge() -> None:
    """Action validity delegates to the graph edge predicate."""
    spec = CounterpointGraphSpec(n=2, tonic=60)
    current_state = (3, 6)
    action = (4, 7)

    assert is_valid_action(current_state, action, spec) == is_valid_edge(
        current_state,
        action,
        spec,
    )


def test_candidate_next_states_returns_only_valid_edges() -> None:
    """Candidate generation returns only states reachable by valid edges."""
    spec = CounterpointGraphSpec(n=2, tonic=60)
    current_state = (3, 6)

    candidates = candidate_next_states(current_state, spec)

    assert candidates
    assert all(is_valid_edge(current_state, candidate, spec) for candidate in candidates)


def test_action_mask_matches_action_space_and_validity() -> None:
    """The action mask mirrors validity over a supplied direct-state action space."""
    spec = CounterpointGraphSpec(n=2, tonic=60)
    current_state = (3, 6)
    action_space = iter_node_states(spec)[:10]

    mask = action_mask(current_state, action_space, spec)

    assert len(mask) == len(action_space)
    assert mask == tuple(
        is_valid_action(current_state, action, spec) for action in action_space
    )
