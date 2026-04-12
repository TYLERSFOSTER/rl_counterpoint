"""Shared symbolic rendering helpers for the music layer."""

from __future__ import annotations

from pathlib import Path

from rl_counterpoint.graph.state_space import ChordState
from rl_counterpoint.music.pitch import midi_to_unicode_note_name


def chord_to_unicode_sequence(chord: ChordState) -> str:
    """Render one chord state as a symbolic note-name sequence."""
    if not chord:
        raise ValueError("chord must not be empty")

    note_names = ", ".join(midi_to_unicode_note_name(midi_note) for midi_note in chord)
    return f"[{note_names}]"


def tonic_meter_to_string(
    *,
    tonic: int,
    measure_size: int,
    target_root_octave: int | None = None,
) -> str:
    """Render tonic and meter as the shared context-conditioning token string."""
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")

    context = f"tonic={midi_to_unicode_note_name(tonic)} meter={measure_size}/4"
    if target_root_octave is not None:
        context += f" target_root_octave={target_root_octave}"

    return context


def _encode_variable_length_quantity(value: int) -> bytes:
    """Encode one MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("value must be non-negative")

    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.append(0x80 | (value & 0x7F))
        value >>= 7

    return bytes(reversed(buffer))


def write_chord_sequence_to_midi(
    *,
    chord_sequence: tuple[ChordState, ...],
    path: Path,
    ticks_per_quarter: int = 480,
    velocity: int = 64,
) -> Path:
    """Write a quarter-note chord sequence to a simple type-0 MIDI file."""
    if not chord_sequence:
        raise ValueError("chord_sequence must not be empty")
    if ticks_per_quarter < 1:
        raise ValueError("ticks_per_quarter must be at least 1")
    if not 0 <= velocity <= 127:
        raise ValueError("velocity must be in [0, 127]")
    if any(not chord for chord in chord_sequence):
        raise ValueError("chord_sequence must not contain empty chords")

    track_events = bytearray()

    for chord in chord_sequence:
        for note_index, midi_note in enumerate(chord):
            if not 0 <= midi_note <= 127:
                raise ValueError("all MIDI notes must be in [0, 127]")
            delta_time = 0 if note_index > 0 else 0
            track_events.extend(_encode_variable_length_quantity(delta_time))
            track_events.extend(bytes((0x90, midi_note, velocity)))

        for note_index, midi_note in enumerate(chord):
            delta_time = ticks_per_quarter if note_index == 0 else 0
            track_events.extend(_encode_variable_length_quantity(delta_time))
            track_events.extend(bytes((0x80, midi_note, 0)))

    track_events.extend(_encode_variable_length_quantity(0))
    track_events.extend(b"\xFF\x2F\x00")

    header = b"MThd" + (6).to_bytes(4, "big")
    header += (0).to_bytes(2, "big")  # format 0
    header += (1).to_bytes(2, "big")  # one track
    header += ticks_per_quarter.to_bytes(2, "big")

    track_chunk = b"MTrk" + len(track_events).to_bytes(4, "big") + bytes(track_events)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + track_chunk)
    return path
