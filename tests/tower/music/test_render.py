"""Tests for tower MIDI rendering helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tower.music.render import (
    encode_variable_length_quantity,
    trajectory_state_sequence,
    write_state_sequence_to_midi,
    write_trajectory_to_midi,
)
from tower.reward.result import TowerRewardResult
from tower.train.trajectory import (
    TRAJECTORY_OUTCOME_VALID,
    TowerTrajectory,
    TowerTrajectoryStep,
)
from tower.window import build_window


def make_rank_1_step(
    *,
    step_index: int,
    source_state: tuple[int, ...],
    realized_next_state: tuple[int, ...],
) -> TowerTrajectoryStep:
    action = (realized_next_state[0] - source_state[0],)
    return TowerTrajectoryStep(
        rank=1,
        step_index=step_index,
        source_state=source_state,
        window=build_window(
            history=(source_state,),
            step_index=step_index,
            measure_size=4,
            context_measures=1,
        ),
        parent_state=None,
        parent_action=None,
        active_choice=action[0],
        assembled_action=action,
        attempted_target_state=realized_next_state,
        realized_next_state=realized_next_state,
        active_logprob=None,
        parent_logprob=None,
        reward=TowerRewardResult(reward=0.0),
        terminated=False,
        truncated=False,
        outcome=TRAJECTORY_OUTCOME_VALID,
    )


def make_rank_2_step(
    *,
    step_index: int,
    source_state: tuple[int, ...],
    realized_next_state: tuple[int, ...],
) -> TowerTrajectoryStep:
    action = (
        realized_next_state[0] - source_state[0],
        realized_next_state[1] - source_state[1],
    )
    return TowerTrajectoryStep(
        rank=2,
        step_index=step_index,
        source_state=source_state,
        window=build_window(
            history=(source_state,),
            step_index=step_index,
            measure_size=4,
            context_measures=1,
        ),
        parent_state=(source_state[0],),
        parent_action=(action[0],),
        active_choice=action[1],
        assembled_action=action,
        attempted_target_state=realized_next_state,
        realized_next_state=realized_next_state,
        active_logprob=None,
        parent_logprob=None,
        reward=TowerRewardResult(reward=0.0),
        terminated=False,
        truncated=False,
        outcome=TRAJECTORY_OUTCOME_VALID,
    )


def test_encode_variable_length_quantity_encodes_known_values() -> None:
    assert encode_variable_length_quantity(0) == b"\x00"
    assert encode_variable_length_quantity(127) == b"\x7f"
    assert encode_variable_length_quantity(128) == b"\x81\x00"
    assert encode_variable_length_quantity(480) == b"\x83\x60"


def test_encode_variable_length_quantity_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="value must be non-negative"):
        encode_variable_length_quantity(-1)


def test_write_state_sequence_to_midi_writes_type_0_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "episode.mid"

    result = write_state_sequence_to_midi(
        state_sequence=((60, 64), (62, 65)),
        path=path,
    )

    data = path.read_bytes()
    assert result == path
    assert data.startswith(b"MThd")
    assert data[8:10] == (0).to_bytes(2, "big")
    assert data[10:12] == (1).to_bytes(2, "big")
    assert data[12:14] == (480).to_bytes(2, "big")
    assert b"MTrk" in data
    assert data.endswith(b"\x00\xff\x2f\x00")


def test_write_state_sequence_to_midi_records_note_events(tmp_path: Path) -> None:
    path = tmp_path / "notes.mid"

    write_state_sequence_to_midi(
        state_sequence=((60, 64),),
        path=path,
        velocity=70,
    )

    data = path.read_bytes()
    assert b"\x00\x90\x3c\x46" in data
    assert b"\x00\x90\x40\x46" in data
    assert b"\x83\x60\x80\x3c\x00" in data
    assert b"\x00\x80\x40\x00" in data


def test_write_state_sequence_to_midi_rejects_empty_sequence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="state_sequence must not be empty"):
        write_state_sequence_to_midi(
            state_sequence=(),
            path=tmp_path / "empty.mid",
        )


def test_write_state_sequence_to_midi_rejects_mixed_rank_states(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="state length must match rank"):
        write_state_sequence_to_midi(
            state_sequence=((60,), (60, 64)),
            path=tmp_path / "mixed.mid",
        )


def test_write_state_sequence_to_midi_rejects_out_of_range_pitch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="MIDI pitch must be in \\[0, 127\\]"):
        write_state_sequence_to_midi(
            state_sequence=((128,),),
            path=tmp_path / "bad.mid",
        )


def test_write_state_sequence_to_midi_rejects_invalid_timing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="ticks_per_quarter must be at least 1"):
        write_state_sequence_to_midi(
            state_sequence=((60,),),
            path=tmp_path / "bad.mid",
            ticks_per_quarter=0,
        )


def test_write_state_sequence_to_midi_rejects_invalid_velocity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="velocity must be in \\[0, 127\\]"):
        write_state_sequence_to_midi(
            state_sequence=((60,),),
            path=tmp_path / "bad.mid",
            velocity=128,
        )


def test_trajectory_state_sequence_includes_initial_and_realized_states() -> None:
    trajectory = TowerTrajectory(
        steps=(
            make_rank_1_step(
                step_index=0,
                source_state=(60,),
                realized_next_state=(61,),
            ),
            make_rank_1_step(
                step_index=1,
                source_state=(61,),
                realized_next_state=(62,),
            ),
        )
    )

    assert trajectory_state_sequence(trajectory) == ((60,), (61,), (62,))


def test_write_trajectory_to_midi_writes_rank_1_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "rank_1.mid"
    trajectory = TowerTrajectory(
        steps=(
            make_rank_1_step(
                step_index=0,
                source_state=(60,),
                realized_next_state=(62,),
            ),
        )
    )

    result = write_trajectory_to_midi(trajectory=trajectory, path=path)

    data = path.read_bytes()
    assert result == path
    assert data.startswith(b"MThd")
    assert b"\x00\x90\x3c\x40" in data
    assert b"\x00\x90\x3e\x40" in data


def test_write_trajectory_to_midi_writes_rank_2_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "rank_2.mid"
    trajectory = TowerTrajectory(
        steps=(
            make_rank_2_step(
                step_index=0,
                source_state=(60, 64),
                realized_next_state=(62, 65),
            ),
        )
    )

    write_trajectory_to_midi(trajectory=trajectory, path=path)

    data = path.read_bytes()
    assert b"\x00\x90\x3c\x40" in data
    assert b"\x00\x90\x40\x40" in data
    assert b"\x00\x90\x3e\x40" in data
    assert b"\x00\x90\x41\x40" in data


def test_write_trajectory_to_midi_rejects_empty_trajectory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trajectory must not be empty"):
        write_trajectory_to_midi(
            trajectory=TowerTrajectory(steps=()),
            path=tmp_path / "empty.mid",
        )
