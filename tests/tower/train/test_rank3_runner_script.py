"""Tests for the dedicated tower rank-3 training script entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import tower_train, tower_train_rank2, tower_train_rank3


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
        ]
    )
    assert exit_code == 0


def test_tower_train_rank3_parse_args_defaults() -> None:
    args = tower_train_rank3.parse_args([])

    assert args.episodes == 1
    assert args.lineage_id == "local-tower"
    assert args.seed == 0
    assert args.max_steps == 1
    assert args.pitch_min == 36
    assert args.pitch_max == 84
    assert args.final_rank == 3
    assert args.initial_pedal_pitch == 62
    assert args.initial_middle_pitch == 65
    assert args.initial_top_pitch == 69
    assert args.sample_initial_state is False
    assert args.initial_parent_pitch_min == 36
    assert args.initial_parent_pitch_max == 84
    assert args.sample_initial_parent_pitch_in_target_octave is False
    assert args.key_pitch_class == 0
    assert args.target_root_octave == 4
    assert args.parent_top_m == 3
    assert args.sample_target_root_octave is False
    assert args.target_root_octave_choices == [2, 3, 4, 5]
    assert args.terminal_cadence_reward == 10.0
    assert args.cadence_failure_reward == 0.0
    assert args.triad_consonance_weight == 1.0
    assert args.triad_non_consonance_penalty == 0.0
    assert args.min_adjacent_gap == 3
    assert args.max_outer_span == 15
    assert args.adjacent_spacing_reward == 0.1
    assert args.adjacent_spacing_penalty == -0.1
    assert args.outer_span_reward == 0.1
    assert args.outer_span_penalty == -0.1
    assert args.cadence_endpoint_weight == 1.0
    assert args.d_model == 32
    assert args.num_layers == 1
    assert args.num_heads == 4
    assert args.ff_dim == 64
    assert args.dropout == 0.0
    assert args.sampling_temperature == 1.5
    assert args.sampling_uniform_mix == 0.15
    assert args.final_inference_sample_target_root_octave is True
    assert args.final_inference_sample_initial_state is True
    assert args.log_reward_diagnostics is True


def test_tower_train_rank3_main_runs_tiny_job(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank2_parent_stack(tmp_path=tmp_path)
    capsys.readouterr()

    exit_code = tower_train_rank3.main(
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
            "--parent-top-m",
            "1",
        ]
    )
    output = capsys.readouterr().out

    run_dir = tmp_path / "lineage-a" / "rank_3"
    assert exit_code == 0
    assert f"run_dir: {run_dir}" in output
    assert "reward: rank3_slice_a" in output
    assert "pitch_range: [36, 84]" in output
    assert "grandparent checkpoint: rank_1/checkpoint_latest.pt" in output
    assert "parent checkpoint: rank_2/checkpoint_latest.pt" in output
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
    assert config["reward_config"]["kind"] == "rank3_slice_a"
    assert config["reward_config"]["key_pitch_class"] == 0
    assert config["reward_config"]["target_root_octave"] == 4
    assert config["reward_config"]["triad_consonance_weight"] == 1.0
    assert config["reward_config"]["triad_non_consonance_penalty"] == 0.0
    assert config["reward_config"]["min_adjacent_gap"] == 3
    assert config["reward_config"]["max_outer_span"] == 15
    assert config["reward_config"]["adjacent_spacing_reward"] == 0.1
    assert config["reward_config"]["adjacent_spacing_penalty"] == -0.1
    assert config["reward_config"]["outer_span_reward"] == 0.1
    assert config["reward_config"]["outer_span_penalty"] == -0.1
    assert config["reward_config"]["cadence_endpoint_weight"] == 1.0
    assert config["parent_sampler_config"]["top_m"] == 1
    assert config["parent_checkpoint"] == "rank_2/checkpoint_latest.pt"
    assert config["graph_config"]["pitch_min"] == 36
    assert config["graph_config"]["pitch_max"] == 84
    assert config["graph_config"]["final_rank"] == 3
    assert config["policy_config"]["d_model"] == 32
    assert config["policy_config"]["num_heads"] == 4
    assert config["policy_config"]["ff_dim"] == 64
    assert config["training_config"]["sampling_temperature"] == 1.5
    assert config["training_config"]["sampling_uniform_mix"] == 0.15
    assert config["training_config"]["sample_target_root_octave"] is False
    assert config["training_config"]["target_root_octave_choices"] == [2, 3, 4, 5]
    assert config["training_config"]["final_inference_sample_target_root_octave"] is True
    assert config["training_config"]["final_inference_sample_initial_state"] is True
    assert config["training_config"]["log_reward_diagnostics"] is True
    diagnostics_rows = (run_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics_rows) == 5


def test_tower_train_rank3_main_can_disable_training_reward_diagnostics(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_rank2_parent_stack(tmp_path=tmp_path)
    capsys.readouterr()

    tower_train_rank3.main(
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
            "--parent-top-m",
            "1",
            "--no-log-reward-diagnostics",
        ]
    )
    capsys.readouterr()

    run_dir = tmp_path / "lineage-a" / "rank_3"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["training_config"]["log_reward_diagnostics"] is False
    assert config["training_config"]["final_inference_sample_target_root_octave"] is True
    assert config["training_config"]["final_inference_sample_initial_state"] is True
    diagnostics_rows = (run_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics_rows) == 4


def test_tower_train_rank3_rejects_missing_parent_stack(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="accepted parent checkpoint is missing"):
        tower_train_rank3.main(
            [
                "--episodes",
                "1",
                "--lineage-id",
                "lineage-a",
                "--artifact-root",
                str(tmp_path),
            ]
        )
