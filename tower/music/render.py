"""MIDI rendering helpers for tower state sequences."""

from __future__ import annotations

from pathlib import Path

from tower.state_action import TowerState, validate_rank
from tower.train.trajectory import TowerTrajectory


def encode_variable_length_quantity(value: int) -> bytes:
    """Encode one MIDI variable-length quantity."""
    if not isinstance(value, int):
        raise TypeError("value must be an int")
    if value < 0:
        raise ValueError("value must be non-negative")

    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.append(0x80 | (value & 0x7F))
        value >>= 7

    return bytes(reversed(buffer))


def write_state_sequence_to_midi(
    *,
    state_sequence: tuple[TowerState, ...],
    path: Path,
    ticks_per_quarter: int = 480,
    velocity: int = 64,
) -> Path:
    """Write a quarter-note tower state sequence to a simple type-0 MIDI file."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not state_sequence:
        raise ValueError("state_sequence must not be empty")
    if ticks_per_quarter < 1:
        raise ValueError("ticks_per_quarter must be at least 1")
    if not 0 <= velocity <= 127:
        raise ValueError("velocity must be in [0, 127]")

    _validate_state_sequence(state_sequence)

    midi_bytes = _midi_file_bytes(
        state_sequence=state_sequence,
        ticks_per_quarter=ticks_per_quarter,
        velocity=velocity,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(midi_bytes)
    return path


def trajectory_state_sequence(trajectory: TowerTrajectory) -> tuple[TowerState, ...]:
    """Return the MIDI state sequence represented by one tower trajectory."""
    if not isinstance(trajectory, TowerTrajectory):
        raise TypeError("trajectory must be a TowerTrajectory")
    if not trajectory.steps:
        raise ValueError("trajectory must not be empty")

    initial_state = trajectory.initial_state
    if initial_state is None:
        raise ValueError("trajectory must not be empty")
    return (initial_state,) + tuple(
        step.realized_next_state for step in trajectory.steps
    )


def write_trajectory_to_midi(
    *,
    trajectory: TowerTrajectory,
    path: Path,
    ticks_per_quarter: int = 480,
    velocity: int = 64,
) -> Path:
    """Write a tower trajectory to a simple type-0 MIDI file."""
    return write_state_sequence_to_midi(
        state_sequence=trajectory_state_sequence(trajectory),
        path=path,
        ticks_per_quarter=ticks_per_quarter,
        velocity=velocity,
    )


def _midi_file_bytes(
    *,
    state_sequence: tuple[TowerState, ...],
    ticks_per_quarter: int,
    velocity: int,
) -> bytes:
    track_events = bytearray()

    for state in state_sequence:
        for midi_note in state:
            delta_time = 0
            track_events.extend(encode_variable_length_quantity(delta_time))
            track_events.extend(bytes((0x90, midi_note, velocity)))

        for note_index, midi_note in enumerate(state):
            delta_time = ticks_per_quarter if note_index == 0 else 0
            track_events.extend(encode_variable_length_quantity(delta_time))
            track_events.extend(bytes((0x80, midi_note, 0)))

    track_events.extend(encode_variable_length_quantity(0))
    track_events.extend(b"\xFF\x2F\x00")

    header = b"MThd" + (6).to_bytes(4, "big")
    header += (0).to_bytes(2, "big")
    header += (1).to_bytes(2, "big")
    header += ticks_per_quarter.to_bytes(2, "big")

    track_chunk = b"MTrk" + len(track_events).to_bytes(4, "big") + bytes(track_events)
    return header + track_chunk


def _validate_state_sequence(state_sequence: tuple[TowerState, ...]) -> None:
    first_state = state_sequence[0]
    if not isinstance(first_state, tuple):
        raise TypeError("state entries must be tuples")
    if not first_state:
        raise ValueError("state entries must not be empty")

    rank = len(first_state)
    validate_rank(rank)
    for state in state_sequence:
        if not isinstance(state, tuple):
            raise TypeError("state entries must be tuples")
        if len(state) != rank:
            raise ValueError("state length must match rank")
        for midi_note in state:
            if not isinstance(midi_note, int):
                raise TypeError("MIDI pitches must be ints")
            if midi_note < 0 or midi_note > 127:
                raise ValueError("MIDI pitch must be in [0, 127]")
