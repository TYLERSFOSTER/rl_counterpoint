"""Shared symbolic rendering helpers for the music layer."""

from __future__ import annotations

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
