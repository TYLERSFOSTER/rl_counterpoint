"""Tests for explicit rank-3 stage scripts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import (
    resume_rank3_stage2,
    tower_train,
    tower_train_rank2,
    tower_train_rank3_stage1,
)


def _prepare_rank2_parent_stack(*, tmp_path: Path) -> None:
    exit_code = tower_train.main(
        [
            "--rank",
            "1",
            "--episodes",
            "1",
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--initial-pitch",
            "62",
            "--no-use-induced-rank1-graph",
            "--no-sample-initial-pitch",
            "--no-sample-target-root-octave",
        ]
    )
    assert exit_code == 0
    exit_code = tower_train_rank2.main(
        [
            "--episodes",
            "1",
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--initial-parent-pitch",
            "62",
            "--initial-child-pitch",
            "69",
            "--parent-top-m",
            "1",
            "--no-sample-key-pitch-class",
            "--no-sample-initial-state",
            "--no-sample-target-root-octave",
            "--no-final-inference-sample-target-root-octave",
            "--no-final-inference-sample-initial-state",
        ]
    )
    assert exit_code == 0


def test_tower_train_rank3_stage1_main_writes_completion_and_heartbeat(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank2_parent_stack(tmp_path=tmp_path)
    capsys.readouterr()

    exit_code = tower_train_rank3_stage1.main(
        [
            "--episodes",
            "1",
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--initial-pedal-pitch",
            "62",
            "--initial-middle-pitch",
            "65",
            "--initial-top-pitch",
            "69",
            "--initial-parent-pitch-min",
            "62",
            "--initial-parent-pitch-max",
            "62",
            "--parent-top-m",
            "1",
            "--no-sample-key-pitch-class",
            "--target-root-octave-choices",
            "4",
        ]
    )
    output = capsys.readouterr().out

    lineage_dir = tmp_path / "lineage-a"
    rank_dir = lineage_dir / "rank_3_stage1"
    assert exit_code == 0
    assert f"rank3 stage1 run_dir: {rank_dir}" in output
    assert (lineage_dir / "heartbeat.json").exists()
    assert (lineage_dir / "completion.json").exists()


def test_resume_rank3_stage2_main_writes_completion_and_examples(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank2_parent_stack(tmp_path=tmp_path)
    capsys.readouterr()
    exit_code = tower_train_rank3_stage1.main(
        [
            "--episodes",
            "1",
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--initial-pedal-pitch",
            "62",
            "--initial-middle-pitch",
            "65",
            "--initial-top-pitch",
            "69",
            "--initial-parent-pitch-min",
            "62",
            "--initial-parent-pitch-max",
            "62",
            "--parent-top-m",
            "1",
            "--no-sample-key-pitch-class",
            "--target-root-octave-choices",
            "4",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    exit_code = resume_rank3_stage2.main(
        [
            "--episodes",
            "1",
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--initial-pedal-pitch",
            "62",
            "--initial-middle-pitch",
            "65",
            "--initial-top-pitch",
            "69",
            "--initial-parent-pitch-min",
            "62",
            "--initial-parent-pitch-max",
            "62",
            "--parent-top-m",
            "1",
            "--no-sample-key-pitch-class",
            "--target-root-octave-choices",
            "4",
        ]
    )
    output = capsys.readouterr().out

    lineage_dir = tmp_path / "lineage-a"
    rank_dir = lineage_dir / "rank_3"
    assert exit_code == 0
    assert f"rank3 stage2 run_dir: {rank_dir}" in output
    assert (rank_dir / "example_episode.mid").exists()
    completion = json.loads((lineage_dir / "completion.json").read_text())
    assert completion["status"] == "completed"
    assert completion["stage"] == "rank3-stage2"
    config = json.loads((rank_dir / "config.json").read_text())
    assert config["training_config"]["sample_initial_state"] is True
    assert (
        config["training_config"]["sample_initial_parent_pitch_in_target_octave"]
        is False
    )
