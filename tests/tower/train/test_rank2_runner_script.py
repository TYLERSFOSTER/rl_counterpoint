"""Tests for the dedicated tower rank-2 training script entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import tower_train, tower_train_rank2

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def test_tower_train_rank2_parse_args_defaults() -> None:
    args = tower_train_rank2.parse_args([])

    assert args.episodes == 1
    assert args.lineage_id == "local-tower"
    assert args.seed == 0
    assert args.max_steps == 1
    assert args.pitch_min == 36
    assert args.pitch_max == 84
    assert args.initial_parent_pitch == 60
    assert args.initial_child_pitch == 64
    assert args.key_pitch_class == 0
    assert args.target_root_octave == 4
    assert args.parent_top_m == 3
    assert args.terminal_cadence_reward == 10.0
    assert args.cadence_failure_reward == 0.0
    assert args.vertical_consonance_weight == 1.0
    assert args.vertical_non_consonance_penalty == -0.5
    assert args.upper_register_soft_ceiling == 80
    assert args.upper_register_penalty_weight == 0.05
    assert args.min_vertical_gap == 3
    assert args.spacing_reward == 0.1
    assert args.spacing_penalty == -0.1
    assert args.d_model == 32
    assert args.num_layers == 1
    assert args.num_heads == 4
    assert args.ff_dim == 64
    assert args.dropout == 0.0
    assert args.sampling_temperature == 1.5
    assert args.sampling_uniform_mix == 0.15
    assert args.log_reward_diagnostics is True


def test_tower_train_rank2_main_runs_tiny_job(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank1_parent(tmp_path=tmp_path)
    capsys.readouterr()

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
            "1",
            "--parent-top-m",
            "1",
        ]
    )
    output = capsys.readouterr().out

    run_dir = tmp_path / "lineage-a" / "rank_2"
    assert exit_code == 0
    assert f"run_dir: {run_dir}" in output
    assert "reward: rank2_slice_a" in output
    assert "pitch_range: [36, 84]" in output
    assert "parent checkpoint: rank_1/checkpoint_latest.pt" in output
    assert "final midi:" in output
    assert "latest checkpoint:" in output
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "reward_diagnostics.jsonl").exists()
    assert (run_dir / "checkpoint_latest.pt").exists()
    assert (run_dir / "example_episode.mid").exists()
    assert (run_dir / "example_episode_1.mid").exists()
    assert (run_dir / "example_episode_2.mid").exists()
    assert (run_dir / "example_episode_3.mid").exists()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["reward_config"]["kind"] == "rank2_slice_a"
    assert config["reward_config"]["key_pitch_class"] == 0
    assert config["reward_config"]["target_root_octave"] == 4
    assert config["reward_config"]["vertical_consonance_weight"] == 1.0
    assert config["reward_config"]["vertical_non_consonance_penalty"] == -0.5
    assert config["reward_config"]["upper_register_soft_ceiling"] == 80
    assert config["reward_config"]["upper_register_penalty_weight"] == 0.05
    assert config["reward_config"]["min_vertical_gap"] == 3
    assert config["reward_config"]["spacing_reward"] == 0.1
    assert config["reward_config"]["spacing_penalty"] == -0.1
    assert config["parent_sampler_config"]["top_m"] == 1
    assert config["parent_checkpoint"] == "rank_1/checkpoint_latest.pt"
    assert config["graph_config"]["pitch_min"] == 36
    assert config["graph_config"]["pitch_max"] == 84
    assert config["policy_config"]["d_model"] == 32
    assert config["policy_config"]["num_heads"] == 4
    assert config["policy_config"]["ff_dim"] == 64
    assert config["training_config"]["sampling_temperature"] == 1.5
    assert config["training_config"]["sampling_uniform_mix"] == 0.15
    assert config["training_config"]["log_reward_diagnostics"] is True
    diagnostics_rows = (run_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics_rows) == 5


def test_tower_train_rank2_main_can_disable_training_reward_diagnostics(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank1_parent(tmp_path=tmp_path)
    capsys.readouterr()

    tower_train_rank2.main(
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
            "--parent-top-m",
            "1",
            "--no-log-reward-diagnostics",
        ]
    )
    capsys.readouterr()

    run_dir = tmp_path / "lineage-a" / "rank_2"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["training_config"]["log_reward_diagnostics"] is False
    diagnostics_rows = (run_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics_rows) == 4


def test_tower_train_rank2_script_runs_by_file_path(tmp_path: Path) -> None:
    _prepare_rank1_parent(tmp_path=tmp_path)
    script_path = PROJECT_ROOT / "scripts" / "tower_train_rank2.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
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
            "--parent-top-m",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / "lineage-a" / "rank_2"
    assert f"run_dir: {run_dir}" in result.stdout
    assert "reward: rank2_slice_a" in result.stdout
    assert "parent checkpoint: rank_1/checkpoint_latest.pt" in result.stdout
    assert "final midi:" in result.stdout
    assert (run_dir / "example_episode.mid").exists()
    assert (run_dir / "example_episode_1.mid").exists()
    assert (run_dir / "example_episode_2.mid").exists()
    assert (run_dir / "example_episode_3.mid").exists()
    assert (run_dir / "reward_diagnostics.jsonl").exists()


def test_tower_train_rank2_rejects_missing_parent(tmp_path: Path) -> None:
    script_path = PROJECT_ROOT / "scripts" / "tower_train_rank2.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--episodes",
            "1",
            "--lineage-id",
            "lineage-a",
            "--artifact-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "accepted parent checkpoint is missing" in result.stderr
