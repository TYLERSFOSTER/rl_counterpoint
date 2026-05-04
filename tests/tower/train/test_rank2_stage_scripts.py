"""Tests for explicit rank-2 stage scripts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import resume_rank2_stage2, tower_train, tower_train_rank2_stage1


def _prepare_rank1_parent(*, tmp_path: Path) -> None:
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
            "1",
        ]
    )
    assert exit_code == 0


def test_tower_train_rank2_stage1_main_writes_completion_and_heartbeat(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank1_parent(tmp_path=tmp_path)
    capsys.readouterr()

    exit_code = tower_train_rank2_stage1.main(
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
            "1",
            "--initial-parent-pitch",
            "40",
            "--initial-child-pitch",
            "43",
            "--initial-parent-pitch-min",
            "40",
            "--initial-parent-pitch-max",
            "40",
            "--parent-top-m",
            "1",
            "--no-sample-key-pitch-class",
            "--target-root-octave-choices",
            "2",
        ]
    )
    output = capsys.readouterr().out

    lineage_dir = tmp_path / "lineage-a"
    rank_dir = lineage_dir / "rank_2_stage1"
    assert exit_code == 0
    assert f"rank2 stage1 run_dir: {rank_dir}" in output
    assert (lineage_dir / "heartbeat.json").exists()
    assert (lineage_dir / "completion.json").exists()


def test_resume_rank2_stage2_main_writes_completion_and_examples(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank1_parent(tmp_path=tmp_path)
    capsys.readouterr()
    exit_code = tower_train_rank2_stage1.main(
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
            "40",
            "--initial-child-pitch",
            "43",
            "--initial-parent-pitch-min",
            "40",
            "--initial-parent-pitch-max",
            "40",
            "--parent-top-m",
            "1",
            "--no-sample-key-pitch-class",
            "--target-root-octave-choices",
            "2",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    exit_code = resume_rank2_stage2.main(
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
            "40",
            "--initial-child-pitch",
            "43",
            "--initial-parent-pitch-min",
            "40",
            "--initial-parent-pitch-max",
            "40",
            "--parent-top-m",
            "1",
            "--no-sample-key-pitch-class",
            "--target-root-octave-choices",
            "2",
        ]
    )
    output = capsys.readouterr().out

    lineage_dir = tmp_path / "lineage-a"
    rank_dir = lineage_dir / "rank_2"
    assert exit_code == 0
    assert f"rank2 stage2 run_dir: {rank_dir}" in output
    assert (rank_dir / "example_episode.mid").exists()
    completion = json.loads((lineage_dir / "completion.json").read_text())
    assert completion["status"] == "completed"
    assert completion["stage"] == "rank2-stage2"
    config = json.loads((rank_dir / "config.json").read_text())
    assert config["training_config"]["sample_initial_state"] is True
    assert (
        config["training_config"]["sample_initial_parent_pitch_in_target_octave"]
        is False
    )
