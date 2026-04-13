"""Tests for the explicit REINFORCE training harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from random import Random

import torch

from scripts import train_reinforce

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_config_binds_eight_measure_episode_cap() -> None:
    """The training harness derives max_steps from the fixed measure cap."""
    config = train_reinforce.TrainConfig()

    assert config.episode_measures == 8
    assert config.measure_size == 4
    assert config.max_steps == 32
    assert config.target_distance_weight == 1.0
    assert config.target_terminal_window_reward == 10.0
    assert config.entropy_coefficient == 0.01
    assert config.export_temperature == 1.0


def test_choose_export_action_index_uses_greedy_argmax_at_zero_temperature() -> None:
    """Zero export temperature preserves deterministic greedy export."""
    action_index = train_reinforce.choose_export_action_index(
        legal_indices=[1, 4, 7],
        legal_logits=torch.tensor([0.5, 2.5, 1.0], dtype=torch.float32),
        export_temperature=0.0,
        rng=Random(123),
    )

    assert action_index == 4


def test_choose_export_action_index_can_sample_non_argmax_action() -> None:
    """Positive export temperature enables stochastic MIDI export."""
    action_index = train_reinforce.choose_export_action_index(
        legal_indices=[10, 11],
        legal_logits=torch.tensor([0.0, 0.0], dtype=torch.float32),
        export_temperature=1.0,
        rng=Random(0),
    )

    assert action_index == 11


def test_append_metrics_writes_jsonl_record(tmp_path: Path) -> None:
    """Episode metrics are appended as JSONL for persistent run logging."""
    metrics_path = train_reinforce.append_metrics(
        run_dir=tmp_path,
        episode_index=2,
        stats=train_reinforce.ReinforceEpisodeStats(
            episode_return=3.0,
            episode_length=32,
            mean_step_reward=0.09375,
            terminated=False,
            truncated=True,
            loss=1.25,
            target_root_octave=4,
            final_root_octave=5,
            final_octave_distance=1,
            hit_target_on_final_step=False,
        ),
    )

    record = json.loads(metrics_path.read_text().strip())
    assert record["episode_index"] == 2
    assert record["episode_return"] == 3.0
    assert record["episode_length"] == 32
    assert record["terminated"] is False
    assert record["truncated"] is True
    assert record["target_root_octave"] == 4
    assert record["final_root_octave"] == 5
    assert record["final_octave_distance"] == 1
    assert record["hit_target_on_final_step"] is False


def test_save_checkpoint_writes_episode_and_latest_files(tmp_path: Path) -> None:
    """The harness persists both per-episode and latest checkpoints."""
    config = train_reinforce.TrainConfig()
    env = train_reinforce.build_env(config)
    policy = train_reinforce.build_policy(env, context_measures=config.context_measures)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    episode_path, latest_path = train_reinforce.save_checkpoint(
        run_dir=tmp_path,
        episode_index=1,
        policy=policy,
        optimizer=optimizer,
        config=config,
        stats=train_reinforce.ReinforceEpisodeStats(
            episode_return=2.0,
            episode_length=32,
            mean_step_reward=0.0625,
            terminated=False,
            truncated=True,
            loss=0.5,
            target_root_octave=4,
            final_root_octave=4,
            final_octave_distance=0,
            hit_target_on_final_step=True,
        ),
    )

    assert episode_path.exists()
    assert latest_path.exists()

    checkpoint = torch.load(latest_path, weights_only=False)
    assert checkpoint["episode_index"] == 1
    assert checkpoint["config"]["episode_measures"] == 8
    assert checkpoint["stats"]["episode_length"] == 32
    assert checkpoint["stats"]["target_root_octave"] == 4
    assert checkpoint["stats"]["final_root_octave"] == 4
    assert checkpoint["stats"]["final_octave_distance"] == 0
    assert checkpoint["stats"]["hit_target_on_final_step"] is True


def test_train_reinforce_main_prints_stats_and_writes_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    """The training entrypoint writes config, metrics, and checkpoints."""
    train_reinforce.main(run_dir=tmp_path)
    output = capsys.readouterr().out

    assert "run_dir:" in output
    assert "episode_measures: 8" in output
    assert "max_steps: 32" in output
    assert "entropy_coefficient: 0.01" in output
    assert "export_temperature: 1.0" in output
    assert "episode 0 return:" in output
    assert "episode 0 mean_step_reward:" in output
    assert "episode 0 target_root_octave:" in output
    assert "episode 0 final_root_octave:" in output
    assert "episode 0 final_octave_distance:" in output
    assert "episode 0 hit_target_on_final_step:" in output
    assert "episode 0 checkpoint:" in output
    assert "latest checkpoint:" in output
    assert "example episode midi:" in output

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert (tmp_path / "example_episode.mid").exists()


def test_train_reinforce_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/train_reinforce.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run_dir:" in result.stdout
    assert "episode_measures: 8" in result.stdout
    assert "max_steps: 32" in result.stdout
    assert "entropy_coefficient: 0.01" in result.stdout
    assert "export_temperature: 1.0" in result.stdout
    assert "episode 0 return:" in result.stdout
    assert "episode 0 target_root_octave:" in result.stdout
    assert "episode 0 checkpoint:" in result.stdout
    assert "example episode midi:" in result.stdout
