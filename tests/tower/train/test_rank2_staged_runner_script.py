"""Tests for the staged tower rank-2 training script entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import tower_train, tower_train_rank2_staged


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


def test_tower_train_rank2_staged_parse_args_defaults() -> None:
    args = tower_train_rank2_staged.parse_args([])

    assert args.stage1_episodes == 5000
    assert args.stage2_episodes == 5000
    assert args.lineage_id == "local-tower-rank2-staged"
    assert args.initial_parent_pitch == 64
    assert args.initial_child_pitch == 68
    assert args.sample_target_root_octave is True
    assert args.target_root_octave_choices == [2, 3, 4, 5]
    assert args.log_reward_diagnostics is False


def test_tower_train_rank2_staged_main_runs_tiny_job(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank1_parent(tmp_path=tmp_path)
    capsys.readouterr()

    exit_code = tower_train_rank2_staged.main(
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
            "--pitch-min",
            "36",
            "--pitch-max",
            "48",
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
            "--target-root-octave-choices",
            "2",
        ]
    )
    output = capsys.readouterr().out

    stage1_dir = tmp_path / "lineage-a" / "rank_2_stage1"
    stage2_dir = tmp_path / "lineage-a" / "rank_2"
    assert exit_code == 0
    assert f"rank2 stage1 run_dir: {stage1_dir}" in output
    assert f"rank2 stage2 run_dir: {stage2_dir}" in output
    assert (stage1_dir / "config.json").exists()
    assert (stage2_dir / "config.json").exists()
    assert (stage2_dir / "example_episode.mid").exists()
    config = json.loads((stage2_dir / "config.json").read_text())
    assert config["training_config"]["sample_initial_state"] is False
    assert config["training_config"]["sample_initial_parent_pitch_in_target_octave"] is False
    assert config["training_config"]["sample_target_root_octave"] is True
    assert config["training_config"]["target_root_octave_choices"] == [2]
