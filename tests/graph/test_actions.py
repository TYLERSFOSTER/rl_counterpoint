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
    candidate_step_delta_actions,
    is_valid_action,
    is_valid_step_delta_action,
    step_delta_action_mask,
    step_delta_action_space,
    step_delta_to_next_state,
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


def test_step_delta_to_next_state_adds_signed_state_changes() -> None:
    """A step delta decodes as per-voice signed change in state."""
    current_state = (3, 6, 10)

    assert step_delta_to_next_state(current_state, (1, -1, 2)) == (4, 5, 12)


def test_step_delta_to_next_state_rejects_wrong_length() -> None:
    """A step delta must bind one state change to each voice."""
    current_state = (3, 6, 10)

    try:
        step_delta_to_next_state(current_state, (1, -1))
    except ValueError as error:
        assert str(error) == "step_delta length must match current_state length"
    else:
        raise AssertionError("expected ValueError")


def test_step_delta_action_space_is_nonzero_bounded_lattice() -> None:
    """The step-delta action space is [-max_step_size, max_step_size]^n minus zero."""
    action_space = step_delta_action_space(n=2, max_step_size=1)

    assert action_space == (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def test_step_delta_action_space_rejects_invalid_bounds() -> None:
    """The step-delta lattice requires at least one voice and positive step size."""
    try:
        step_delta_action_space(n=0, max_step_size=1)
    except ValueError as error:
        assert str(error) == "n must be at least 1"
    else:
        raise AssertionError("expected ValueError")

    try:
        step_delta_action_space(n=2, max_step_size=0)
    except ValueError as error:
        assert str(error) == "max_step_size must be at least 1"
    else:
        raise AssertionError("expected ValueError")


def test_is_valid_step_delta_action_delegates_decoded_target_to_edge_predicate() -> None:
    """Step-delta validity is edge validity after target decoding."""
    spec = CounterpointGraphSpec(n=2, tonic=60)
    current_state = (3, 6)
    step_delta = (1, 1)
    target = step_delta_to_next_state(current_state, step_delta)

    assert is_valid_step_delta_action(current_state, step_delta, spec) == is_valid_edge(
        current_state,
        target,
        spec,
    )


def test_candidate_step_delta_actions_return_only_valid_edges() -> None:
    """Candidate step deltas decode only to valid graph edges."""
    spec = CounterpointGraphSpec(n=2, tonic=60)
    current_state = (3, 6)

    candidates = candidate_step_delta_actions(
        current_state,
        spec,
        max_step_size=2,
    )

    assert candidates
    assert all(
        is_valid_edge(
            current_state,
            step_delta_to_next_state(current_state, candidate),
            spec,
        )
        for candidate in candidates
    )


def test_step_delta_action_mask_matches_action_space_and_validity() -> None:
    """The step-delta mask mirrors validity over a fixed delta lattice."""
    spec = CounterpointGraphSpec(n=2, tonic=60)
    current_state = (3, 6)
    action_space = step_delta_action_space(n=spec.n, max_step_size=1)

    mask = step_delta_action_mask(current_state, action_space, spec)

    assert len(mask) == len(action_space)
    assert mask == tuple(
        is_valid_step_delta_action(current_state, step_delta, spec)
        for step_delta in action_space
    )
