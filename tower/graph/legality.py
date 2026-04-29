"""Minimal graph legality checks for early tower slices."""

from __future__ import annotations

from tower.graph.spec import TowerGraphSpec
from tower.graph.projection import project_action, project_state
from tower.state_action import TowerAction, TowerState, apply_action, validate_action, validate_state

RANK2_ALLOWED_VERTICAL_INTERVAL_CLASSES = frozenset({3, 4, 6, 7, 8, 9, 10})
RANK2_MAX_VERTICAL_INTERVAL = 10
RANK1_ALLOWED_DIATONIC_INTERVAL_CLASSES = frozenset({0, 2, 4, 5, 7, 9, 11})
RANK3_ALLOWED_ADJACENT_INTERVAL_CLASSES = frozenset({3, 4, 7, 8, 9})
RANK3_ALLOWED_OUTER_INTERVAL_CLASSES = frozenset({6, 7, 8, 10})
RANK3_MAX_OUTER_INTERVAL = 15


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


def _rank3_adjacent_intervals(state: TowerState) -> tuple[int, int]:
    return (state[1] - state[0], state[2] - state[1])


def _rank3_outer_interval(state: TowerState) -> int:
    return state[2] - state[0]


def _rank3_pair_indices() -> tuple[tuple[int, int], ...]:
    return ((0, 1), (1, 2), (0, 2))


def _rank3_has_voice_crossing(source: TowerState, target: TowerState) -> bool:
    return (
        target[0] >= source[1]
        or source[0] >= target[1]
        or target[1] >= source[2]
        or source[1] >= target[2]
    )


def _rank3_has_stationary_voice(action: TowerAction) -> bool:
    return any(delta == 0 for delta in action)


def _rank3_has_parallel_perfect_interval(
    source: TowerState,
    target: TowerState,
    *,
    interval_class: int,
) -> bool:
    for lower_index, upper_index in _rank3_pair_indices():
        source_interval = source[upper_index] - source[lower_index]
        displacement_lower = target[lower_index] - source[lower_index]
        displacement_upper = target[upper_index] - source[upper_index]
        if (
            source_interval % 12 == interval_class
            and displacement_lower == displacement_upper
        ):
            return True
    return False


def is_valid_state(state: TowerState, spec: TowerGraphSpec) -> bool:
    """Return True iff a state satisfies the minimal rank/range contract."""
    try:
        validate_state(state, rank=spec.rank)
    except (TypeError, ValueError):
        return False

    if not all(spec.pitch_min <= pitch <= spec.pitch_max for pitch in state):
        return False

    if spec.rank == 1:
        pitch_class = (state[0] - spec.key_pitch_class) % 12
        if pitch_class not in RANK1_ALLOWED_DIATONIC_INTERVAL_CLASSES:
            return False
        if spec.induced_node_image is not None:
            return state in spec.induced_node_image

    if spec.rank == 2:
        projected_parent_spec = TowerGraphSpec(
            rank=1,
            key_pitch_class=spec.key_pitch_class,
            pitch_min=spec.pitch_min,
            pitch_max=spec.pitch_max,
            max_step_size=spec.max_step_size,
        )
        if not is_valid_state((state[0],), projected_parent_spec):
            return False
        vertical_interval = _rank2_vertical_interval(state)
        if vertical_interval > RANK2_MAX_VERTICAL_INTERVAL:
            return False
        if vertical_interval % 12 not in RANK2_ALLOWED_VERTICAL_INTERVAL_CLASSES:
            return False
        if spec.induced_node_image is not None:
            return state in spec.induced_node_image

    if spec.rank == 3:
        projected_rank1_spec = TowerGraphSpec(
            rank=1,
            key_pitch_class=spec.key_pitch_class,
            pitch_min=spec.pitch_min,
            pitch_max=spec.pitch_max,
            max_step_size=spec.max_step_size,
        )
        if not is_valid_state((state[0],), projected_rank1_spec):
            return False

        projected_rank2_spec = TowerGraphSpec(
            rank=2,
            key_pitch_class=spec.key_pitch_class,
            pitch_min=spec.pitch_min,
            pitch_max=spec.pitch_max,
            max_step_size=spec.max_step_size,
        )
        if not is_valid_state(project_state(state), projected_rank2_spec):
            return False

        lower_adjacent_interval, upper_adjacent_interval = _rank3_adjacent_intervals(state)
        if lower_adjacent_interval % 12 not in RANK3_ALLOWED_ADJACENT_INTERVAL_CLASSES:
            return False
        if upper_adjacent_interval % 12 not in RANK3_ALLOWED_ADJACENT_INTERVAL_CLASSES:
            return False

        outer_interval = _rank3_outer_interval(state)
        if outer_interval > RANK3_MAX_OUTER_INTERVAL:
            return False
        if outer_interval % 12 not in RANK3_ALLOWED_OUTER_INTERVAL_CLASSES:
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

    if spec.rank in {1, 2} and spec.induced_edge_image is not None:
        return (source, target) in spec.induced_edge_image and is_valid_state(target, spec)

    if spec.rank == 2:
        if _rank2_has_stationary_voice(action):
            return False
        if _rank2_has_voice_crossing(source, target):
            return False
        if _rank2_has_parallel_fifth(source, target):
            return False

    if spec.rank == 3:
        if _rank3_has_stationary_voice(action):
            return False
        if _rank3_has_voice_crossing(source, target):
            return False
        if _rank3_has_parallel_perfect_interval(source, target, interval_class=7):
            return False
        if _rank3_has_parallel_perfect_interval(source, target, interval_class=0):
            return False
        projected_rank2_spec = TowerGraphSpec(
            rank=2,
            key_pitch_class=spec.key_pitch_class,
            pitch_min=spec.pitch_min,
            pitch_max=spec.pitch_max,
            max_step_size=spec.max_step_size,
        )
        if not is_valid_transition(
            project_state(source),
            project_action(action),
            projected_rank2_spec,
        ):
            return False

    return is_valid_state(target, spec)
