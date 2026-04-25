"""Minimal graph legality checks for early tower slices."""

from __future__ import annotations

from tower.graph.spec import TowerGraphSpec
from tower.state_action import TowerAction, TowerState, apply_action, validate_action, validate_state

RANK2_ALLOWED_VERTICAL_INTERVAL_CLASSES = frozenset({3, 4, 5, 7, 8, 9})
RANK2_ALLOWED_ROOT_INTERVAL_CLASSES = frozenset({3, 4, 5, 7, 8, 9})
RANK2_MAX_VERTICAL_INTERVAL = 10


def _rank2_vertical_interval(state: TowerState) -> int:
    return state[1] - state[0]


def _rank2_has_voice_crossing(source: TowerState, target: TowerState) -> bool:
    return target[0] >= source[1] or source[0] >= target[1]


def _rank2_has_parallel_fifth(source: TowerState, target: TowerState) -> bool:
    return (
        _rank2_vertical_interval(source) == 7
        and target[0] - source[0] == target[1] - source[1]
    )


def _rank2_has_stationary_voice(action: TowerAction) -> bool:
    return any(delta == 0 for delta in action)


def is_valid_state(state: TowerState, spec: TowerGraphSpec) -> bool:
    """Return True iff a state satisfies the minimal rank/range contract."""
    try:
        validate_state(state, rank=spec.rank)
    except (TypeError, ValueError):
        return False

    if not all(spec.pitch_min <= pitch <= spec.pitch_max for pitch in state):
        return False

    if spec.rank == 1 and spec.induced_node_image is not None:
        return state in spec.induced_node_image

    if spec.rank == 2:
        lower_pitch_class = (state[0] - spec.key_pitch_class) % 12
        if lower_pitch_class not in RANK2_ALLOWED_ROOT_INTERVAL_CLASSES:
            return False
        vertical_interval = _rank2_vertical_interval(state)
        if vertical_interval > RANK2_MAX_VERTICAL_INTERVAL:
            return False
        if vertical_interval % 12 not in RANK2_ALLOWED_VERTICAL_INTERVAL_CLASSES:
            return False

    return True


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

    if spec.rank == 1 and spec.induced_edge_image is not None:
        return (source, target) in spec.induced_edge_image

    if spec.rank == 2:
        if _rank2_has_stationary_voice(action):
            return False
        if _rank2_has_voice_crossing(source, target):
            return False
        if _rank2_has_parallel_fifth(source, target):
            return False

    return is_valid_state(target, spec)
