"""Minimal graph legality checks for early tower slices."""

from __future__ import annotations

from tower.graph.spec import TowerGraphSpec
from tower.state_action import TowerAction, TowerState, apply_action, validate_action, validate_state


def is_valid_state(state: TowerState, spec: TowerGraphSpec) -> bool:
    """Return True iff a state satisfies the minimal rank/range contract."""
    try:
        validate_state(state, rank=spec.rank)
    except (TypeError, ValueError):
        return False

    return all(spec.pitch_min <= pitch <= spec.pitch_max for pitch in state)


def is_valid_transition(
    source: TowerState,
    action: TowerAction,
    spec: TowerGraphSpec,
) -> bool:
    """Return True iff action carries source to a minimally valid non-self target."""
    if not is_valid_state(source, spec):
        return False

    try:
        validate_action(action, rank=spec.rank)
        target = apply_action(source, action)
    except (TypeError, ValueError):
        return False

    if target == source:
        return False

    return is_valid_state(target, spec)
