"""Tests for the staged tower training script entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import tower_train_staged

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_tower_train_staged_parse_args_defaults() -> None:
    args = tower_train_staged.parse_args([])

    assert args.lineage_id == "local-tower-staged"
    assert args.stage1_episodes == 3_000
    assert args.stage2_episodes == 3_000
    assert args.max_steps == 64
    assert args.max_step_size == 7
    assert args.pitch_min == 0
    assert args.pitch_max == 127
    assert args.use_induced_rank1_graph is True
    assert args.induced_rank2_pitch_min == 0
    assert args.induced_rank2_pitch_max == 127
    assert args.induced_rank2_max_step_size == 7
    assert args.target_root_octave_choices == [2, 3, 4, 5]
    assert args.goal_octave_direction_weight == 0.5
    assert args.terminal_cadence_reward == 100.0
    assert args.d_model == 64
    assert args.num_layers == 2
    assert args.ff_dim == 128
    assert args.progress_every == 250
    assert args.sampling_temperature == 1.5
    assert args.sampling_uniform_mix == 0.15
    assert args.log_reward_diagnostics is False


def test_tower_train_staged_main_runs_tiny_two_stage_job(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = tower_train_staged.main(
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
            "2,3,4",
            "--progress-every",
            "1",
        ]
    )
    output = capsys.readouterr().out

    stage1_run_dir = tmp_path / "lineage-a-stage1" / "rank_1"
    stage2_run_dir = tmp_path / "lineage-a" / "rank_1"
    assert exit_code == 0
    assert "[progress] stage1 1/1" in output
    assert "[progress] stage2 1/1" in output
    assert f"stage1 run_dir: {stage1_run_dir}" in output
    assert f"stage2 run_dir: {stage2_run_dir}" in output
    assert (stage1_run_dir / "config.json").exists()
    assert (stage1_run_dir / "checkpoint_latest.pt").exists()
    assert (stage2_run_dir / "config.json").exists()
    assert (stage2_run_dir / "checkpoint_latest.pt").exists()
    assert (stage2_run_dir / "example_episode.mid").exists()

    stage1_config = json.loads((stage1_run_dir / "config.json").read_text())
    stage2_config = json.loads((stage2_run_dir / "config.json").read_text())
    assert (
        stage1_config["training_config"]["sample_initial_pitch_in_target_octave"]
        is True
    )
    assert (
        stage2_config["training_config"]["sample_initial_pitch_in_target_octave"]
        is False
    )
    assert stage2_config["policy_config"]["d_model"] == 64
    assert stage2_config["graph_config"]["pitch_min"] == 0
    assert stage2_config["graph_config"]["pitch_max"] == 127
    assert stage2_config["graph_config"]["use_induced_rank1_graph"] is False
    assert stage2_config["graph_config"]["induced_rank2_pitch_min"] == 0
    assert stage2_config["graph_config"]["induced_rank2_pitch_max"] == 127
    assert stage2_config["graph_config"]["induced_rank2_max_step_size"] == 7
    assert stage2_config["policy_config"]["num_layers"] == 2
    assert stage2_config["policy_config"]["ff_dim"] == 128
    assert stage2_config["reward_config"]["terminal_cadence_reward"] == 100.0
    assert stage2_config["reward_config"]["goal_octave_direction_weight"] == 0.5
    assert stage2_config["training_config"]["target_root_octave_choices"] == [2, 3, 4]
    assert stage2_config["training_config"]["progress_every"] == 1
    assert stage2_config["training_config"]["sampling_temperature"] == 1.5
    assert stage2_config["training_config"]["sampling_uniform_mix"] == 0.15


def test_tower_train_staged_script_runs_by_file_path(tmp_path: Path) -> None:
    script_path = PROJECT_ROOT / "scripts" / "tower_train_staged.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
            "--stage1-episodes",
            "1",
            "--stage2-episodes",
            "1",
            "--max-steps",
            "1",
            "--max-step-size",
            "2",
            "--no-use-induced-rank1-graph",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stage2_run_dir = tmp_path / "lineage-a" / "rank_1"
    assert f"stage2 run_dir: {stage2_run_dir}" in result.stdout
    assert (stage2_run_dir / "example_episode.mid").exists()


def test_tower_train_staged_main_writes_exception_log_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    def boom(**_: object):
        raise RuntimeError("forced crash for test")

    monkeypatch.setattr(tower_train_staged, "run_rank1_training", boom)

    with pytest.raises(RuntimeError, match="forced crash for test"):
        tower_train_staged.main(
            [
                "--lineage-id",
                "lineage-crash",
                "--artifact-root",
                str(tmp_path),
                "--stage1-episodes",
                "1",
                "--stage2-episodes",
                "1",
            ]
        )

    captured = capsys.readouterr()
    exception_log_path = tmp_path / "logs" / "lineage-crash.exception.log"
    assert f"exception log: {exception_log_path}" in captured.err
    assert exception_log_path.exists()
    exception_log = exception_log_path.read_text()
    assert "RuntimeError: forced crash for test" in exception_log
    assert "lineage_id: lineage-crash" in exception_log
