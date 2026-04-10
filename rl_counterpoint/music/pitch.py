"""Shared pitch helpers for the music layer."""

from __future__ import annotations

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def pitch_class(pitch: int) -> int:
    """Return the pitch class of one integer pitch."""
    return pitch % 12


def midi_to_unicode_note_name(midi_note: int) -> str:
    """Map a MIDI note value to the project's unicode-style note token."""
    if midi_note < 0 or midi_note > 127:
        raise ValueError("midi_note must be in [0, 127]")

    note_name = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{note_name}_{octave}"
