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

    assert config.episode_measures == 16
    assert config.measure_size == 4
    assert config.max_steps == 64
    assert config.max_step_size == 4
    assert config.num_episodes == 100_000
    assert config.learning_rate == 1e-3
    assert config.gamma == 0.9
    assert config.target_distance_weight == 1.0
    assert config.target_terminal_window_reward == 10.0
    assert config.entropy_coefficient == 0.01
    assert config.epsilon_behavior == 0.2
    assert config.goal_bias_weight == 0.0
    assert config.export_temperature == 1.0
    assert config.beat_role_consonance_weight == 1.0
    assert config.early_goal_weight == 2.0


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


def test_save_checkpoint_writes_latest_file_only(tmp_path: Path) -> None:
    """The harness persists only the rolling latest checkpoint."""
    config = train_reinforce.TrainConfig()
    env = train_reinforce.build_env(config)
    policy = train_reinforce.build_policy(env, context_measures=config.context_measures)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    latest_path = train_reinforce.save_checkpoint(
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

    assert latest_path.exists()
    assert latest_path.name == "checkpoint_latest.pt"
    assert not any(tmp_path.glob("checkpoint_episode_*.pt"))

    checkpoint = torch.load(latest_path, weights_only=False)
    assert checkpoint["episode_index"] == 1
    assert checkpoint["config"]["episode_measures"] == 16
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
    original_train_config = train_reinforce.TrainConfig

    @train_reinforce.dataclass(frozen=True)
    class ShortTrainConfig(original_train_config):
        num_episodes: int = 2

    train_reinforce.TrainConfig = ShortTrainConfig
    try:
        train_reinforce.main(run_dir=tmp_path)
    finally:
        train_reinforce.TrainConfig = original_train_config
    output = capsys.readouterr().out

    assert "run_dir:" in output
    assert "episode_measures: 16" in output
    assert "max_steps: 64" in output
    assert "entropy_coefficient: 0.01" in output
    assert "epsilon_behavior: 0.2" in output
    assert "goal_bias_weight: 0.0" in output
    assert "export_temperature: 1.0" in output
    assert "beat_role_consonance_weight: 1.0" in output
    assert "early_goal_weight: 2.0" in output
    assert "episode 0 return:" in output
    assert "episode 0 mean_step_reward:" in output
    assert "episode 0 target_root_octave:" in output
    assert "episode 0 final_root_octave:" in output
    assert "episode 0 final_octave_distance:" in output
    assert "episode 0 hit_target_on_final_step:" in output
    assert "latest checkpoint:" in output
    assert "example episode midi:" in output

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert (tmp_path / "example_episode.mid").exists()


def test_train_reinforce_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    script_path = tmp_path = PROJECT_ROOT / "scripts" / "train_reinforce.py"
    script_text = script_path.read_text()
    patched_text = script_text.replace(
        "DEFAULT_NUM_EPISODES = 100_000",
        "DEFAULT_NUM_EPISODES = 2",
        1,
    )
    temp_script = PROJECT_ROOT / "scripts" / "_tmp_train_reinforce_test.py"
    temp_script.write_text(patched_text)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        temp_script.unlink(missing_ok=True)

    assert "run_dir:" in result.stdout
    assert "episode_measures: 16" in result.stdout
    assert "max_steps: 64" in result.stdout
    assert "entropy_coefficient: 0.01" in result.stdout
    assert "epsilon_behavior: 0.2" in result.stdout
    assert "goal_bias_weight: 0.0" in result.stdout
    assert "export_temperature: 1.0" in result.stdout
    assert "beat_role_consonance_weight: 1.0" in result.stdout
    assert "early_goal_weight: 2.0" in result.stdout
    assert "episode 0 return:" in result.stdout
    assert "episode 0 target_root_octave:" in result.stdout
    assert "example episode midi:" in result.stdout
