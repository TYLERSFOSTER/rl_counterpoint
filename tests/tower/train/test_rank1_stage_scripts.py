"""Tests for explicit rank-1 stage scripts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import resume_rank1_stage2, tower_train_rank1_stage1


def test_tower_train_rank1_stage1_main_writes_completion_and_heartbeat(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = tower_train_rank1_stage1.main(
        [
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--stage1-episodes",
            "1",
            "--stage2-episodes",
            "1",
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--no-use-induced-rank1-graph",
            "--target-root-octave-choices",
            "2",
        ]
    )
    output = capsys.readouterr().out

    lineage_dir = tmp_path / "lineage-a-stage1"
    rank_dir = lineage_dir / "rank_1"
    assert exit_code == 0
    assert f"stage1 run_dir: {rank_dir}" in output
    assert (lineage_dir / "heartbeat.json").exists()
    assert (lineage_dir / "completion.json").exists()


def test_resume_rank1_stage2_main_writes_completion_and_examples(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = tower_train_rank1_stage1.main(
        [
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--stage1-episodes",
            "1",
            "--stage2-episodes",
            "1",
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--no-use-induced-rank1-graph",
            "--target-root-octave-choices",
            "2",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    exit_code = resume_rank1_stage2.main(
        [
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--stage1-episodes",
            "1",
            "--stage2-episodes",
            "1",
            "--seed",
            "123",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--no-use-induced-rank1-graph",
            "--target-root-octave-choices",
            "2",
        ]
    )
    output = capsys.readouterr().out

    lineage_dir = tmp_path / "lineage-a"
    rank_dir = lineage_dir / "rank_1"
    assert exit_code == 0
    assert f"stage2 run_dir: {rank_dir}" in output
    assert (rank_dir / "example_episode.mid").exists()
    completion = json.loads((lineage_dir / "completion.json").read_text())
    assert completion["status"] == "completed"
    assert completion["stage"] == "rank1-stage2"
