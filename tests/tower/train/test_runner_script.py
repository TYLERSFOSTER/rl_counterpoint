"""Tests for the tower training script entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import tower_train

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_tower_train_parse_args_defaults_to_rank_1() -> None:
    args = tower_train.parse_args([])

    assert args.rank == 1
    assert args.episodes == 1
    assert args.lineage_id == "local-tower"
    assert args.max_steps == 1
    assert args.key_pitch_class == 0
    assert args.target_root_octave == 4
    assert args.terminal_cadence_reward == 10.0
    assert args.range_penalty == -1.0


def test_tower_train_main_runs_tiny_rank_1_job(
    tmp_path: Path,
    capsys,
) -> None:
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
    output = capsys.readouterr().out

    run_dir = tmp_path / "lineage-a" / "rank_1"
    assert exit_code == 0
    assert f"run_dir: {run_dir}" in output
    assert "reward: rank1_slice_a" in output
    assert "final midi:" in output
    assert "latest checkpoint:" in output
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "reward_diagnostics.jsonl").exists()
    assert (run_dir / "checkpoint_latest.pt").exists()
    assert (run_dir / "example_episode.mid").exists()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["reward_config"]["kind"] == "rank1_slice_a"
    assert config["reward_config"]["key_pitch_class"] == 0
    assert config["reward_config"]["target_root_octave"] == 4
    diagnostics_rows = (run_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics_rows) == 2


def test_tower_train_script_runs_by_file_path(tmp_path: Path) -> None:
    script_path = PROJECT_ROOT / "scripts" / "tower_train.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
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
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / "lineage-a" / "rank_1"
    assert f"run_dir: {run_dir}" in result.stdout
    assert "reward: rank1_slice_a" in result.stdout
    assert "final midi:" in result.stdout
    assert (run_dir / "example_episode.mid").exists()
    assert (run_dir / "reward_diagnostics.jsonl").exists()


def test_tower_train_script_rejects_rank_2_until_parent_loading_exists(
    tmp_path: Path,
) -> None:
    script_path = PROJECT_ROOT / "scripts" / "tower_train.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--rank",
            "2",
            "--episodes",
            "1",
            "--artifact-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "currently supports rank 1 only" in result.stderr
