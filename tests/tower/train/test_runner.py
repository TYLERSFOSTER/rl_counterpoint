"""Tests for tower training runner contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tower.graph.spec import TowerGraphSpec
from tower.policy.base import PolicyOutput
from tower.reward.factory import build_rank1_reward_fn
from tower.reward.result import TowerRewardResult
from tower.train.checkpoint import (
    DEFAULT_TOWER_ARTIFACT_ROOT,
    TowerArtifactPaths,
    build_checkpoint_payload,
    load_latest_checkpoint,
    read_lineage_manifest,
    read_rank_config,
    read_rank_metrics,
    read_reward_diagnostics,
    record_rank_manifest_entry,
    save_latest_checkpoint,
)
from tower.train.config import TowerRankConfig
from tower.train.runner import (
    FinalInferenceResult,
    Rank1TrainingRunResult,
    Rank2TrainingRunResult,
    TowerRunnerConfig,
    run_rank1_training,
    run_rank2_training,
    run_final_inference_episode,
)
from tower.window import TowerWindow


class TinyRank1Policy(torch.nn.Module):
    rank = 1

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert len(state) == 1
        assert window.valid_mask[-1]
        return PolicyOutput(logits=self.logits)


class TinyRank2Policy(torch.nn.Module):
    rank = 2

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 2.0, 0.0]))

    def forward(
        self,
        *,
        state: tuple[int, ...],
        window: TowerWindow,
    ) -> PolicyOutput:
        assert len(state) == 2
        assert window.valid_mask[-1]
        return PolicyOutput(logits=self.logits)


def prepare_accepted_rank1_parent(
    *,
    tmp_path: Path,
    lineage_id: str = "lineage-a",
) -> TowerArtifactPaths:
    rank_1_paths = TowerArtifactPaths(
        lineage_id=lineage_id,
        rank=1,
        artifact_root=tmp_path,
    )
    rank_1_config = TowerRankConfig(
        rank=1,
        lineage_id=lineage_id,
        episode_budget=1,
        max_step_size=1,
    )
    save_latest_checkpoint(
        paths=rank_1_paths,
        payload=build_checkpoint_payload(
            config=rank_1_config,
            episode_index=0,
            stats={"episode_return": 1.0},
            policy_state_dict={"parent_weight": [1.0]},
            optimizer_state_dict={"step": 1},
        ),
    )
    record_rank_manifest_entry(paths=rank_1_paths, status="accepted")
    return rank_1_paths


def test_runner_config_records_rank_1_run_settings() -> None:
    config = TowerRunnerConfig(
        lineage_id="lineage-a",
        rank=1,
        episode_count=5,
        seed=123,
    )

    assert config.lineage_id == "lineage-a"
    assert config.rank == 1
    assert config.episode_count == 5
    assert config.seed == 123
    assert config.artifact_root == DEFAULT_TOWER_ARTIFACT_ROOT
    assert config.parent_checkpoint is None
    assert config.parent_top_m == 3
    assert config.final_midi_enabled is True


def test_runner_config_builds_artifact_paths(tmp_path: Path) -> None:
    config = TowerRunnerConfig(
        lineage_id="lineage-a",
        rank=1,
        episode_count=5,
        seed=123,
        artifact_root=tmp_path,
    )

    paths = config.artifact_paths()

    assert paths.lineage_id == "lineage-a"
    assert paths.rank == 1
    assert paths.artifact_root == tmp_path
    assert paths.rank_dir == tmp_path / "lineage-a" / "rank_1"


def test_runner_config_converts_to_rank_config() -> None:
    config = TowerRunnerConfig(
        lineage_id="lineage-a",
        rank=1,
        episode_count=5,
        seed=123,
        measure_size=3,
        context_measures=4,
        max_step_size=2,
        reward_config={"terminal": 10.0},
        graph_config={"pitch_min": 48},
        policy_config={"d_model": 16},
        training_config={"gamma": 0.9},
    )

    rank_config = config.to_rank_config()

    assert isinstance(rank_config, TowerRankConfig)
    assert rank_config.rank == 1
    assert rank_config.lineage_id == "lineage-a"
    assert rank_config.episode_budget == 5
    assert rank_config.measure_size == 3
    assert rank_config.context_measures == 4
    assert rank_config.max_step_size == 2
    assert rank_config.reward_config == {"terminal": 10.0}
    assert rank_config.graph_config == {"pitch_min": 48}
    assert rank_config.policy_config == {"d_model": 16}
    assert rank_config.training_config == {"gamma": 0.9, "episode_count": 5}
    assert rank_config.parent_sampler_config == {}
    assert rank_config.parent_checkpoint is None
    assert rank_config.seed_config == {"seed": 123}


def test_rank_2_runner_config_requires_parent_checkpoint() -> None:
    with pytest.raises(ValueError, match="rank greater than 1 requires"):
        TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=2,
            episode_count=5,
            seed=123,
        )


def test_rank_2_runner_config_records_parent_top_m_default() -> None:
    config = TowerRunnerConfig(
        lineage_id="lineage-a",
        rank=2,
        episode_count=5,
        seed=123,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
    )

    rank_config = config.to_rank_config()

    assert config.parent_top_m == 3
    assert rank_config.parent_checkpoint == "rank_1/checkpoint_latest.pt"
    assert rank_config.parent_sampler_config == {"top_m": 3}


def test_rank_2_runner_config_allows_parent_top_m_override() -> None:
    config = TowerRunnerConfig(
        lineage_id="lineage-a",
        rank=2,
        episode_count=5,
        seed=123,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
        parent_top_m=2,
    )

    assert config.to_rank_config().parent_sampler_config == {"top_m": 2}


def test_rank_1_runner_config_rejects_parent_checkpoint() -> None:
    with pytest.raises(ValueError, match="rank 1 runner config must not"):
        TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=5,
            seed=123,
            parent_checkpoint="rank_0/checkpoint_latest.pt",
        )


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    (
        ("lineage_id", "", "lineage_id must not be empty"),
        ("episode_count", 0, "episode_count must be at least 1"),
        ("seed", -1, "seed must be non-negative"),
        ("measure_size", 0, "measure_size must be at least 1"),
        ("context_measures", 0, "context_measures must be at least 1"),
        ("max_step_size", 0, "max_step_size must be at least 1"),
        ("parent_top_m", 0, "parent_top_m must be at least 1"),
    ),
)
def test_runner_config_rejects_invalid_values(
    field_name: str,
    value: object,
    error: str,
) -> None:
    kwargs = {
        "lineage_id": "lineage-a",
        "rank": 1,
        "episode_count": 5,
        "seed": 123,
        field_name: value,
    }

    with pytest.raises(ValueError, match=error):
        TowerRunnerConfig(**kwargs)  # type: ignore[arg-type]


def test_runner_config_rejects_non_path_artifact_root() -> None:
    with pytest.raises(TypeError, match="artifact_root must be a Path"):
        TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=5,
            seed=123,
            artifact_root="artifacts/tower",  # type: ignore[arg-type]
        )


def test_runner_config_rejects_non_bool_final_midi_enabled() -> None:
    with pytest.raises(TypeError, match="final_midi_enabled must be a bool"):
        TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=5,
            seed=123,
            final_midi_enabled=1,  # type: ignore[arg-type]
        )


def test_runner_config_rejects_non_json_config_values() -> None:
    with pytest.raises(TypeError, match="policy_config.bad must be JSON-compatible"):
        TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=5,
            seed=123,
            policy_config={"bad": object()},
        )


def test_final_inference_runs_rank_1_episode_without_gradients() -> None:
    policy = TinyRank1Policy()
    policy.train()

    result = run_final_inference_episode(
        policy=policy,
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
        generator=torch.Generator().manual_seed(0),
    )

    assert isinstance(result, FinalInferenceResult)
    assert result.trajectory.rank == 1
    assert len(result.trajectory.steps) == 1
    assert result.metrics["rank"] == 1
    assert result.metrics["episode_return"] == 1.0
    assert result.metrics["final_inference"] is True
    assert result.metrics["final_state"] == result.trajectory.final_state
    assert "loss" not in result.metrics
    assert not policy.training
    assert policy.logits.grad is None
    active_logprob = result.trajectory.steps[0].active_logprob
    assert isinstance(active_logprob, torch.Tensor)
    assert active_logprob.requires_grad is False


def test_final_inference_records_terminal_success_metrics() -> None:
    policy = TinyRank1Policy()

    result = run_final_inference_episode(
        policy=policy,
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(
            reward=2.0,
            is_terminal_success=True,
        ),
        max_steps=3,
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.metrics["episode_length"] == 1
    assert result.metrics["mean_step_reward"] == 2.0
    assert result.metrics["terminated"] is True
    assert result.metrics["truncated"] is False
    assert result.metrics["terminal_success"] is True


def test_final_inference_runs_rank_2_with_parent_policy_without_gradients() -> None:
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    parent_policy.train()
    child_policy.train()

    result = run_final_inference_episode(
        policy=child_policy,
        parent_policy=parent_policy,
        initial_state=(60, 64),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        max_steps=1,
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        parent_top_m=1,
        generator=torch.Generator().manual_seed(0),
    )

    step = result.trajectory.steps[0]
    assert result.trajectory.rank == 2
    assert step.parent_logprob is not None
    assert step.active_logprob is not None
    assert isinstance(step.parent_logprob, torch.Tensor)
    assert isinstance(step.active_logprob, torch.Tensor)
    assert step.parent_logprob.requires_grad is False
    assert step.active_logprob.requires_grad is False
    assert parent_policy.logits.grad is None
    assert child_policy.logits.grad is None
    assert not parent_policy.training
    assert not child_policy.training
    assert result.metrics["rank"] == 2
    assert result.metrics["episode_return"] == 1.0


def test_final_inference_rejects_rank_1_parent_policy() -> None:
    with pytest.raises(ValueError, match="rank 1 final inference must not"):
        run_final_inference_episode(
            policy=TinyRank1Policy(),
            parent_policy=TinyRank1Policy(),
            initial_state=(60,),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
        )


def test_final_inference_rejects_rank_2_without_parent_policy() -> None:
    with pytest.raises(ValueError, match="rank 2 final inference requires parent"):
        run_final_inference_episode(
            policy=TinyRank2Policy(),
            initial_state=(60, 64),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        )


def test_final_inference_rejects_graph_spec_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="graph spec rank must match"):
        run_final_inference_episode(
            policy=TinyRank1Policy(),
            initial_state=(60,),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
            max_steps=1,
            graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        )


def test_run_rank1_training_writes_artifacts_and_final_midi(tmp_path: Path) -> None:
    policy = TinyRank1Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    config = TowerRunnerConfig(
        lineage_id="lineage-a",
        rank=1,
        episode_count=2,
        seed=123,
        artifact_root=tmp_path,
        max_step_size=1,
        training_config={"max_steps": 1, "gamma": 1.0},
    )

    result = run_rank1_training(
        config=config,
        policy=policy,
        optimizer=optimizer,
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
    )

    assert isinstance(result, Rank1TrainingRunResult)
    assert result.paths.rank_dir == tmp_path / "lineage-a" / "rank_1"
    assert result.rank_config == read_rank_config(result.paths)
    assert len(result.episode_results) == 2
    assert len(result.final_inferences) == 4
    assert len(result.final_midi_paths) == 4
    assert result.final_midi_path == result.paths.example_episode_path
    assert result.paths.example_episode_path.exists()
    assert result.paths.example_episode_path.read_bytes().startswith(b"MThd")
    for index in range(1, 4):
        midi_path = result.paths.rank_dir / f"example_episode_{index}.mid"
        assert midi_path.exists()
        assert midi_path.read_bytes().startswith(b"MThd")
    assert result.paths.reward_diagnostics_path.exists()

    metrics = read_rank_metrics(result.paths)
    assert len(metrics) == 6
    assert metrics[0]["episode_index"] == 0
    assert metrics[1]["episode_index"] == 1
    assert [row["kind"] for row in metrics[2:]] == ["final_inference"] * 4
    assert [row["final_inference_index"] for row in metrics[2:]] == [0, 1, 2, 3]
    assert metrics[2]["midi_path"] == "rank_1/example_episode.mid"
    assert metrics[3]["midi_path"] == "rank_1/example_episode_1.mid"
    assert metrics[5]["midi_path"] == "rank_1/example_episode_3.mid"
    assert metrics[2]["final_inference"] is True

    diagnostics = read_reward_diagnostics(result.paths)
    assert len(diagnostics) == 6
    assert [row["episode_index"] for row in diagnostics] == [0, 1, 2, 3, 4, 5]
    assert diagnostics[-1]["episode_kind"] == "final_inference"
    assert diagnostics[-1]["reward"] == 1.0

    checkpoint = load_latest_checkpoint(result.paths)
    assert checkpoint["rank"] == 1
    assert checkpoint["lineage_id"] == "lineage-a"
    assert checkpoint["episode_index"] == 1
    assert checkpoint["config"] == result.rank_config.to_json_dict()

    manifest = read_lineage_manifest(result.paths)
    assert manifest["ranks"]["1"]["status"] == "accepted"
    assert (
        manifest["ranks"]["1"]["reward_diagnostics"]
        == "rank_1/reward_diagnostics.jsonl"
    )


def test_run_rank1_training_can_sample_episode_initial_pitch_and_target(
    tmp_path: Path,
) -> None:
    policy = TinyRank1Policy()
    result = run_rank1_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=8,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            final_midi_enabled=False,
            reward_config={
                "target_root_octave": 4,
                "use_context_target_root_octave": True,
            },
            training_config={
                "max_steps": 1,
                "sample_initial_pitch": True,
                "initial_pitch_min": 50,
                "initial_pitch_max": 52,
                "sample_target_root_octave": True,
                "target_root_octave_choices": [2, 3],
            },
        ),
        policy=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
    )

    metrics = read_rank_metrics(result.paths)
    training_rows = [row for row in metrics if not row.get("final_inference")]
    initial_pitches = {row["initial_state"][0] for row in training_rows}
    target_octaves = {row["target_root_octave"] for row in training_rows}
    assert initial_pitches <= {50, 51, 52}
    assert target_octaves <= {2, 3}
    assert len(initial_pitches) > 1
    assert len(target_octaves) > 1


def test_run_rank1_training_optimizer_changes_policy(tmp_path: Path) -> None:
    policy = TinyRank1Policy()
    before = policy.logits.detach().clone()

    run_rank1_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=1,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            training_config={"max_steps": 1},
        ),
        policy=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
    )

    assert not torch.allclose(policy.logits.detach(), before)


def test_run_rank1_training_can_build_default_transformer_policy(
    tmp_path: Path,
) -> None:
    result = run_rank1_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=1,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            policy_config={
                "d_model": 8,
                "num_layers": 1,
                "num_heads": 2,
                "ff_dim": 16,
                "dropout": 0.0,
            },
            training_config={"max_steps": 1},
        ),
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
    )

    assert result.policy.rank == 1
    assert len(result.episode_results) == 1
    assert len(result.final_inferences) == 4
    assert result.final_midi_path == result.paths.example_episode_path
    assert result.paths.example_episode_path.exists()


def test_run_rank1_training_can_skip_final_midi(tmp_path: Path) -> None:
    result = run_rank1_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=1,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            final_midi_enabled=False,
            training_config={"max_steps": 1},
        ),
        policy=TinyRank1Policy(),
        initial_state=(60,),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
    )

    metrics = read_rank_metrics(result.paths)
    assert result.final_midi_path is None
    assert result.final_midi_paths == (None, None, None, None)
    assert not result.paths.example_episode_path.exists()
    assert all(row["midi_path"] is None for row in metrics[-4:])
    assert result.paths.reward_diagnostics_path.exists()


def test_run_rank1_training_accepts_rank1_reward_factory(tmp_path: Path) -> None:
    result = run_rank1_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=1,
            episode_count=1,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            final_midi_enabled=False,
            training_config={"max_steps": 1},
        ),
        policy=TinyRank1Policy(),
        initial_state=(60,),
        reward_fn=build_rank1_reward_fn(),
        graph_spec=TowerGraphSpec(rank=1, max_step_size=1),
    )

    assert len(result.episode_results) == 1
    reward_diagnostics = result.episode_results[0].trajectory.steps[0].reward.diagnostics
    assert reward_diagnostics["kind"] == "rank1_reward"
    artifact_diagnostics = read_reward_diagnostics(result.paths)
    assert artifact_diagnostics[0]["reward_diagnostics"]["kind"] == "rank1_reward"


def test_run_rank1_training_rejects_non_rank_1_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rank-1 runner config"):
        run_rank1_training(
            config=TowerRunnerConfig(
                lineage_id="lineage-a",
                rank=2,
                episode_count=1,
                seed=123,
                artifact_root=tmp_path,
                parent_checkpoint="rank_1/checkpoint_latest.pt",
            ),
            policy=TinyRank1Policy(),
            initial_state=(60,),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
        )


def test_run_rank2_training_writes_parent_linked_artifacts_and_final_midi(
    tmp_path: Path,
) -> None:
    prepare_accepted_rank1_parent(tmp_path=tmp_path)
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_optimizer = torch.optim.SGD(child_policy.parameters(), lr=0.1)
    config = TowerRunnerConfig(
        lineage_id="lineage-a",
        rank=2,
        episode_count=2,
        seed=123,
        artifact_root=tmp_path,
        max_step_size=1,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
        parent_top_m=1,
        training_config={"max_steps": 1, "gamma": 1.0},
    )

    result = run_rank2_training(
        config=config,
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=child_optimizer,
        initial_state=(60, 64),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
    )

    assert isinstance(result, Rank2TrainingRunResult)
    assert result.paths.rank_dir == tmp_path / "lineage-a" / "rank_2"
    assert result.rank_config == read_rank_config(result.paths)
    assert len(result.episode_results) == 2
    assert len(result.final_inferences) == 4
    assert len(result.final_midi_paths) == 4
    assert result.final_midi_path == result.paths.example_episode_path
    assert result.paths.example_episode_path.exists()
    assert result.paths.example_episode_path.read_bytes().startswith(b"MThd")
    for index in range(1, 4):
        midi_path = result.paths.rank_dir / f"example_episode_{index}.mid"
        assert midi_path.exists()
        assert midi_path.read_bytes().startswith(b"MThd")
    assert result.paths.reward_diagnostics_path.exists()

    metrics = read_rank_metrics(result.paths)
    assert len(metrics) == 6
    assert metrics[0]["episode_index"] == 0
    assert metrics[1]["episode_index"] == 1
    assert [row["kind"] for row in metrics[2:]] == ["final_inference"] * 4
    assert [row["final_inference_index"] for row in metrics[2:]] == [0, 1, 2, 3]
    assert metrics[2]["midi_path"] == "rank_2/example_episode.mid"
    assert metrics[3]["midi_path"] == "rank_2/example_episode_1.mid"
    assert metrics[2]["final_inference"] is True
    assert metrics[2]["rank"] == 2

    diagnostics = read_reward_diagnostics(result.paths)
    assert len(diagnostics) == 6
    assert diagnostics[0]["rank"] == 2
    assert diagnostics[0]["parent_state"] == [60]
    assert diagnostics[0]["parent_action"] == [1]
    assert diagnostics[-1]["episode_kind"] == "final_inference"

    checkpoint = load_latest_checkpoint(result.paths)
    assert checkpoint["rank"] == 2
    assert checkpoint["lineage_id"] == "lineage-a"
    assert checkpoint["episode_index"] == 1
    assert checkpoint["parent_rank"] == 1
    assert checkpoint["parent_checkpoint"] == "rank_1/checkpoint_latest.pt"
    assert checkpoint["parent_artifact_schema_version"] == 1
    assert "parent_policy_state_dict" not in checkpoint

    manifest = read_lineage_manifest(result.paths)
    rank_2_entry = manifest["ranks"]["2"]
    assert rank_2_entry["status"] == "accepted"
    assert rank_2_entry["parent_rank"] == 1
    assert rank_2_entry["parent_checkpoint"] == "rank_1/checkpoint_latest.pt"
    assert (
        rank_2_entry["reward_diagnostics"]
        == "rank_2/reward_diagnostics.jsonl"
    )


def test_run_rank2_training_child_optimizer_changes_child_policy(
    tmp_path: Path,
) -> None:
    prepare_accepted_rank1_parent(tmp_path=tmp_path)
    parent_policy = TinyRank1Policy()
    child_policy = TinyRank2Policy()
    child_before = child_policy.logits.detach().clone()

    run_rank2_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=2,
            episode_count=1,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            parent_checkpoint="rank_1/checkpoint_latest.pt",
            parent_top_m=1,
            training_config={"max_steps": 1},
        ),
        parent_policy=parent_policy,
        child_policy=child_policy,
        child_optimizer=torch.optim.SGD(child_policy.parameters(), lr=0.1),
        initial_state=(60, 64),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
    )

    assert not torch.allclose(child_policy.logits.detach(), child_before)


def test_run_rank2_training_preserves_parent_policy_and_checkpoint(
    tmp_path: Path,
) -> None:
    rank_1_paths = prepare_accepted_rank1_parent(tmp_path=tmp_path)
    parent_checkpoint_bytes = rank_1_paths.checkpoint_latest_path.read_bytes()
    parent_checkpoint_payload = load_latest_checkpoint(rank_1_paths)
    parent_policy = TinyRank1Policy()
    parent_before = parent_policy.logits.detach().clone()

    run_rank2_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=2,
            episode_count=1,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            parent_checkpoint="rank_1/checkpoint_latest.pt",
            parent_top_m=1,
            training_config={"max_steps": 1},
        ),
        parent_policy=parent_policy,
        child_policy=TinyRank2Policy(),
        initial_state=(60, 64),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
    )

    assert torch.equal(parent_policy.logits.detach(), parent_before)
    assert parent_policy.logits.grad is None
    assert all(not parameter.requires_grad for parameter in parent_policy.parameters())
    assert rank_1_paths.checkpoint_latest_path.read_bytes() == parent_checkpoint_bytes
    assert load_latest_checkpoint(rank_1_paths) == parent_checkpoint_payload


def test_run_rank2_training_can_build_default_child_transformer_policy(
    tmp_path: Path,
) -> None:
    prepare_accepted_rank1_parent(tmp_path=tmp_path)

    result = run_rank2_training(
        config=TowerRunnerConfig(
            lineage_id="lineage-a",
            rank=2,
            episode_count=1,
            seed=123,
            artifact_root=tmp_path,
            max_step_size=1,
            parent_checkpoint="rank_1/checkpoint_latest.pt",
            parent_top_m=1,
            policy_config={
                "d_model": 8,
                "num_layers": 1,
                "num_heads": 2,
                "ff_dim": 16,
                "dropout": 0.0,
            },
            training_config={"max_steps": 1},
        ),
        parent_policy=TinyRank1Policy(),
        initial_state=(60, 64),
        reward_fn=lambda context: TowerRewardResult(reward=1.0),
        graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
    )

    assert result.child_policy.rank == 2
    assert len(result.episode_results) == 1
    assert len(result.final_inferences) == 4
    assert result.final_midi_path == result.paths.example_episode_path
    assert result.paths.example_episode_path.exists()


def test_run_rank2_training_rejects_missing_accepted_parent(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="accepted parent checkpoint is missing"):
        run_rank2_training(
            config=TowerRunnerConfig(
                lineage_id="lineage-a",
                rank=2,
                episode_count=1,
                seed=123,
                artifact_root=tmp_path,
                max_step_size=1,
                parent_checkpoint="rank_1/checkpoint_latest.pt",
                parent_top_m=1,
                training_config={"max_steps": 1},
            ),
            parent_policy=TinyRank1Policy(),
            child_policy=TinyRank2Policy(),
            initial_state=(60, 64),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
            graph_spec=TowerGraphSpec(rank=2, max_step_size=1),
        )


def test_run_rank2_training_rejects_non_rank_2_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rank-2 runner config"):
        run_rank2_training(
            config=TowerRunnerConfig(
                lineage_id="lineage-a",
                rank=1,
                episode_count=1,
                seed=123,
                artifact_root=tmp_path,
            ),
            parent_policy=TinyRank1Policy(),
            initial_state=(60, 64),
            reward_fn=lambda context: TowerRewardResult(reward=1.0),
        )
