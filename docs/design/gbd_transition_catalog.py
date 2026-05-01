"""Generate nearby diatonic three-note transitions out of G-B-D in C major.

Assumption:
- "maximum range to whole steps in each voice" is interpreted as
  "each voice may move by at most two whole steps", i.e. 4 semitones.

When run, this script:
1. Enumerates every reachable three-note diatonic voicing in C major
   from the source chord G-B-D.
2. Requires all three voices to move (no stationary notes).
3. Excludes target sonorities containing a vertical minor second or tritone.
4. Writes a MIDI catalog beside this script so each transition can be heard.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_counterpoint.music.pitch import midi_to_unicode_note_name
from rl_counterpoint.graph.non_crossing import has_voice_crossing

TICKS_PER_QUARTER = 480
VELOCITY = 64
MIDI_CHANNEL = 0
PROGRAM_NUMBER = 0  # Acoustic Grand Piano

SOURCE_CHORD = (67, 71, 74)  # G4, B4, D5
KEY_PITCH_CLASSES = frozenset({0, 2, 4, 5, 7, 9, 11})  # C major
MAX_VOICE_DELTA = 4  # two whole steps
OUTPUT_NAME = "gbd_transition_catalog.mid"


def is_c_major_pitch(pitch: int) -> bool:
    return (pitch % 12) in KEY_PITCH_CLASSES


def has_forbidden_vertical_interval(state: tuple[int, int, int]) -> bool:
    for i, lower in enumerate(state):
        for upper in state[i + 1 :]:
            interval_class = (upper - lower) % 12
            if interval_class in {1, 6}:
                return True
    return False


def is_allowed_diatonic_voicing(state: tuple[int, int, int]) -> bool:
    if not (state[0] < state[1] < state[2]):
        return False
    if not all(is_c_major_pitch(pitch) for pitch in state):
        return False
    if has_forbidden_vertical_interval(state):
        return False
    return True


def generate_transitions() -> tuple[tuple[int, int, int], ...]:
    transitions: list[tuple[int, int, int]] = []
    deltas = tuple(delta for delta in range(-MAX_VOICE_DELTA, MAX_VOICE_DELTA + 1) if delta != 0)

    for lower_delta in deltas:
        for middle_delta in deltas:
            for upper_delta in deltas:
                target = (
                    SOURCE_CHORD[0] + lower_delta,
                    SOURCE_CHORD[1] + middle_delta,
                    SOURCE_CHORD[2] + upper_delta,
                )
                if not is_allowed_diatonic_voicing(target):
                    continue
                if has_voice_crossing(SOURCE_CHORD, target):
                    continue
                transitions.append(target)

    return tuple(sorted(set(transitions)))


def chord_name(chord: tuple[int, int, int]) -> str:
    return "[" + ", ".join(midi_to_unicode_note_name(pitch) for pitch in chord) + "]"


def transition_catalog_sequence(
    transitions: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    states: list[tuple[int, int, int]] = []
    for target in transitions:
        states.append(SOURCE_CHORD)
        states.append(target)
    return tuple(states)


def encode_variable_length_quantity(value: int) -> bytes:
    if value < 0:
        raise ValueError("value must be non-negative")

    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.append(0x80 | (value & 0x7F))
        value >>= 7

    return bytes(reversed(buffer))


def write_state_sequence_to_two_four_midi(
    *,
    state_sequence: tuple[tuple[int, int, int], ...],
    path: Path,
) -> Path:
    track_events = bytearray()

    # Track name.
    track_name = b"GBD transition catalog"
    track_events.extend(encode_variable_length_quantity(0))
    track_events.extend(b"\xFF\x03" + bytes((len(track_name),)) + track_name)

    # Time signature: 2/4.
    track_events.extend(encode_variable_length_quantity(0))
    track_events.extend(b"\xFF\x58\x04\x02\x02\x18\x08")

    # Explicitly pin the whole file to one instrument on one channel.
    track_events.extend(encode_variable_length_quantity(0))
    track_events.extend(bytes((0xC0 | MIDI_CHANNEL, PROGRAM_NUMBER)))

    for state in state_sequence:
        for midi_note in state:
            track_events.extend(encode_variable_length_quantity(0))
            track_events.extend(bytes((0x90 | MIDI_CHANNEL, midi_note, VELOCITY)))

        for note_index, midi_note in enumerate(state):
            delta_time = TICKS_PER_QUARTER if note_index == 0 else 0
            track_events.extend(encode_variable_length_quantity(delta_time))
            track_events.extend(bytes((0x80 | MIDI_CHANNEL, midi_note, 0)))

    track_events.extend(encode_variable_length_quantity(0))
    track_events.extend(b"\xFF\x2F\x00")

    header = b"MThd" + (6).to_bytes(4, "big")
    header += (0).to_bytes(2, "big")
    header += (1).to_bytes(2, "big")
    header += TICKS_PER_QUARTER.to_bytes(2, "big")

    track_chunk = b"MTrk" + len(track_events).to_bytes(4, "big") + bytes(track_events)
    path.write_bytes(header + track_chunk)
    return path


def main() -> None:
    transitions = generate_transitions()
    if not transitions:
        raise RuntimeError("No transitions found under the current constraints.")

    script_dir = Path(__file__).resolve().parent
    midi_path = script_dir / OUTPUT_NAME
    write_state_sequence_to_two_four_midi(
        state_sequence=transition_catalog_sequence(transitions),
        path=midi_path,
    )

    print(f"source: {chord_name(SOURCE_CHORD)}")
    print(f"count: {len(transitions)}")
    for index, target in enumerate(transitions, start=1):
        print(f"{index:03d}. {chord_name(SOURCE_CHORD)} -> {chord_name(target)}")
    print(f"midi: {midi_path}")


if __name__ == "__main__":
    main()
