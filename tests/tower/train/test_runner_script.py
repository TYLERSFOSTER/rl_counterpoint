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
    assert args.pitch_min == 0
    assert args.pitch_max == 127
    assert args.use_induced_rank1_graph is True
    assert args.induced_rank2_pitch_min == 0
    assert args.induced_rank2_pitch_max == 127
    assert args.induced_rank2_max_step_size == 1
    assert args.sample_initial_pitch is True
    assert args.initial_pitch_min == 36
    assert args.initial_pitch_max == 84
    assert args.key_pitch_class == 0
    assert args.target_root_octave == 4
    assert args.goal_octave_direction_weight == 0.5
    assert args.sample_target_root_octave is True
    assert args.target_root_octave_choices is None
    assert args.sample_initial_pitch_in_target_octave is False
    assert args.terminal_cadence_reward == 10.0
    assert args.range_penalty == -1.0
    assert args.measure_start_tonic_reward == 1.0
    assert args.onbeat_scale_degree_reward == 1.0
    assert args.offbeat_consonance_weight == 1.0
    assert args.onbeat_non_scale_penalty == -2.0
    assert args.offbeat_non_consonance_penalty == -2.0
    assert args.step_size_balance_threshold == 3
    assert args.step_size_balance_target_small_rate == 0.3
    assert args.step_size_balance_weight == 1.0
    assert args.sampling_temperature == 1.5
    assert args.sampling_uniform_mix == 0.15
    assert args.log_reward_diagnostics is True


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
            "--no-use-induced-rank1-graph",
        ]
    )
    output = capsys.readouterr().out

    run_dir = tmp_path / "lineage-a" / "rank_1"
    assert exit_code == 0
    assert f"run_dir: {run_dir}" in output
    assert "pitch_range: [0, 127]" in output
    assert "reward: rank1_slice_a" in output
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
    assert config["reward_config"]["kind"] == "rank1_slice_a"
    assert config["reward_config"]["key_pitch_class"] == 0
    assert config["reward_config"]["target_root_octave"] == 4
    assert config["reward_config"]["use_context_target_root_octave"] is True
    assert config["reward_config"]["goal_octave_direction_weight"] == 0.5
    assert config["reward_config"]["measure_start_tonic_reward"] == 1.0
    assert config["reward_config"]["onbeat_scale_degree_reward"] == 1.0
    assert config["reward_config"]["offbeat_consonance_weight"] == 1.0
    assert config["reward_config"]["onbeat_non_scale_penalty"] == -2.0
    assert config["reward_config"]["offbeat_non_consonance_penalty"] == -2.0
    assert config["reward_config"]["step_size_balance_threshold"] == 3
    assert config["reward_config"]["step_size_balance_target_small_rate"] == 0.3
    assert config["reward_config"]["step_size_balance_weight"] == 1.0
    assert config["policy_config"]["d_model"] == 32
    assert config["graph_config"]["pitch_min"] == 0
    assert config["graph_config"]["pitch_max"] == 127
    assert config["graph_config"]["use_induced_rank1_graph"] is False
    assert config["graph_config"]["induced_rank2_pitch_min"] == 0
    assert config["graph_config"]["induced_rank2_pitch_max"] == 127
    assert config["graph_config"]["induced_rank2_max_step_size"] == 1
    assert config["policy_config"]["num_heads"] == 4
    assert config["policy_config"]["ff_dim"] == 64
    assert config["training_config"]["sample_initial_pitch"] is True
    assert config["training_config"]["initial_pitch_min"] == 36
    assert config["training_config"]["initial_pitch_max"] == 84
    assert config["training_config"]["sample_initial_pitch_in_target_octave"] is False
    assert config["training_config"]["sample_target_root_octave"] is True
    assert config["training_config"]["sampling_temperature"] == 1.5
    assert config["training_config"]["sampling_uniform_mix"] == 0.15
    assert config["training_config"]["log_reward_diagnostics"] is True
    diagnostics_rows = (run_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics_rows) == 5


def test_tower_train_main_writes_target_octave_choices(
    tmp_path: Path,
    capsys,
) -> None:
    tower_train.main(
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
            "--no-use-induced-rank1-graph",
            "--target-root-octave-choices",
            "2,3,4",
            "--sample-initial-pitch-in-target-octave",
        ]
    )
    capsys.readouterr()

    config = json.loads(
        (tmp_path / "lineage-a" / "rank_1" / "config.json").read_text()
    )
    assert config["training_config"]["target_root_octave_choices"] == [2, 3, 4]
    assert config["training_config"]["sample_initial_pitch_in_target_octave"] is True


def test_tower_train_main_can_disable_training_reward_diagnostics(
    tmp_path: Path,
    capsys,
) -> None:
    tower_train.main(
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
            "--no-use-induced-rank1-graph",
            "--no-log-reward-diagnostics",
        ]
    )
    capsys.readouterr()

    run_dir = tmp_path / "lineage-a" / "rank_1"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["training_config"]["log_reward_diagnostics"] is False
    diagnostics_rows = (run_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics_rows) == 4


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
            "--no-use-induced-rank1-graph",
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
    assert (run_dir / "example_episode_1.mid").exists()
    assert (run_dir / "example_episode_2.mid").exists()
    assert (run_dir / "example_episode_3.mid").exists()
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
