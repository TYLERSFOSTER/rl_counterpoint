"""Tests for tower artifact path contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tower.train.checkpoint import (
    CHECKPOINT_LATEST_FILENAME,
    CONFIG_FILENAME,
    DEFAULT_TOWER_ARTIFACT_ROOT,
    EXAMPLE_EPISODE_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_STATUSES,
    METRICS_FILENAME,
    TowerArtifactPaths,
    append_rank_metrics,
    build_checkpoint_payload,
    find_accepted_parent_checkpoint,
    load_accepted_parent_checkpoint,
    load_latest_checkpoint,
    read_lineage_manifest,
    read_rank_config,
    read_rank_metrics,
    record_rank_manifest_entry,
    save_latest_checkpoint,
    write_lineage_manifest,
    write_rank_config,
)
from tower.train.config import TowerRankConfig


def test_lineage_and_rank_paths_are_deterministic() -> None:
    paths = TowerArtifactPaths(lineage_id="lineage-a", rank=2)

    assert paths.artifact_root == DEFAULT_TOWER_ARTIFACT_ROOT
    assert paths.lineage_dir == Path("artifacts") / "tower" / "lineage-a"
    assert paths.rank_dir == Path("artifacts") / "tower" / "lineage-a" / "rank_2"


def test_standard_rank_file_paths_are_deterministic() -> None:
    paths = TowerArtifactPaths(lineage_id="lineage-a", rank=1)

    assert paths.manifest_path == paths.lineage_dir / MANIFEST_FILENAME
    assert paths.config_path == paths.rank_dir / CONFIG_FILENAME
    assert paths.metrics_path == paths.rank_dir / METRICS_FILENAME
    assert paths.checkpoint_latest_path == paths.rank_dir / CHECKPOINT_LATEST_FILENAME
    assert paths.example_episode_path == paths.rank_dir / EXAMPLE_EPISODE_FILENAME


def test_relative_manifest_paths_are_deterministic() -> None:
    paths = TowerArtifactPaths(lineage_id="lineage-a", rank=3)

    assert paths.relative_config_path == Path("rank_3") / "config.json"
    assert paths.relative_metrics_path == Path("rank_3") / "metrics.jsonl"
    assert (
        paths.relative_checkpoint_latest_path
        == Path("rank_3") / "checkpoint_latest.pt"
    )
    assert paths.relative_example_episode_path == Path("rank_3") / "example_episode.mid"


def test_custom_artifact_root_supported(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    assert paths.lineage_dir == tmp_path / "lineage-a"
    assert paths.rank_dir == tmp_path / "lineage-a" / "rank_1"


def test_paths_do_not_create_files_or_directories(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    assert not paths.lineage_dir.exists()
    assert not paths.rank_dir.exists()
    assert not paths.config_path.exists()


def test_paths_reject_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank must be at least 1"):
        TowerArtifactPaths(lineage_id="lineage-a", rank=0)


def test_paths_reject_empty_lineage_id() -> None:
    with pytest.raises(ValueError, match="lineage_id must not be empty"):
        TowerArtifactPaths(lineage_id="", rank=1)


def test_paths_reject_non_path_artifact_root() -> None:
    with pytest.raises(TypeError, match="artifact_root must be a Path"):
        TowerArtifactPaths(
            lineage_id="lineage-a",
            rank=1,
            artifact_root="artifacts/tower",  # type: ignore[arg-type]
        )


def test_write_rank_config_creates_rank_dir_and_writes_config(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=10,
    )

    written_path = write_rank_config(config=config, paths=paths)

    assert written_path == paths.config_path
    assert paths.rank_dir.exists()
    assert paths.config_path.exists()


def test_read_rank_config_round_trips_written_config(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=2,
        lineage_id="lineage-a",
        episode_budget=20,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
        parent_sampler_config={"top_m": 3},
    )

    write_rank_config(config=config, paths=paths)

    assert read_rank_config(paths) == config
    assert read_rank_config(paths.config_path) == config


def test_write_rank_config_uses_deterministic_json_format(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=10,
        reward_config={"z": 1, "a": 2},
    )

    write_rank_config(config=config, paths=paths)

    text = paths.config_path.read_text()
    assert text.endswith("\n")
    assert text.splitlines()[0] == "{"
    assert '  "artifact_schema_version": 1,' in text
    assert text.index('"artifact_schema_version"') < text.index('"context_measures"')


def test_write_rank_config_rejects_rank_mismatch(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=10,
    )

    with pytest.raises(ValueError, match="config rank must match"):
        write_rank_config(config=config, paths=paths)


def test_write_rank_config_rejects_lineage_mismatch(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-b",
        episode_budget=10,
    )

    with pytest.raises(ValueError, match="config lineage_id must match"):
        write_rank_config(config=config, paths=paths)


def test_append_rank_metrics_creates_rank_dir_and_metrics_file(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    written_path = append_rank_metrics(
        paths=paths,
        metrics={"rank": 1, "episode_index": 0, "episode_return": 1.5},
    )

    assert written_path == paths.metrics_path
    assert paths.rank_dir.exists()
    assert paths.metrics_path.exists()


def test_append_rank_metrics_writes_jsonl_rows_in_order(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )

    append_rank_metrics(
        paths=paths,
        metrics={"rank": 2, "episode_index": 0, "episode_return": 1.0},
    )
    append_rank_metrics(
        paths=paths,
        metrics={"rank": 2, "episode_index": 1, "episode_return": -0.5},
    )

    assert read_rank_metrics(paths) == (
        {"episode_index": 0, "episode_return": 1.0, "rank": 2},
        {"episode_index": 1, "episode_return": -0.5, "rank": 2},
    )


def test_read_rank_metrics_accepts_path_and_missing_file_returns_empty(
    tmp_path: Path,
) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    assert read_rank_metrics(paths) == ()
    assert read_rank_metrics(paths.metrics_path) == ()


def test_append_rank_metrics_rejects_non_json_value(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    with pytest.raises(TypeError, match="metrics.bad must be JSON-compatible"):
        append_rank_metrics(
            paths=paths,
            metrics={"rank": 1, "bad": object()},
        )


def test_append_rank_metrics_rejects_rank_mismatch(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    with pytest.raises(ValueError, match="metrics rank must match"):
        append_rank_metrics(
            paths=paths,
            metrics={"rank": 2, "episode_index": 0},
        )


def test_append_rank_metrics_rejects_non_int_episode_index(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    with pytest.raises(TypeError, match="metrics episode_index must be an int"):
        append_rank_metrics(
            paths=paths,
            metrics={"rank": 1, "episode_index": "0"},
        )


def test_append_rank_metrics_accepts_core_slice_6_fields(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    metrics = {
        "rank": 2,
        "episode_index": 3,
        "episode_return": 4.5,
        "episode_length": 16,
        "mean_step_reward": 0.25,
        "terminated": True,
        "truncated": False,
        "loss": 0.125,
        "invalid_extension_count": 1,
        "empty_lift_fiber_count": 0,
        "parent_failure_count": 0,
        "terminal_success": True,
    }

    append_rank_metrics(paths=paths, metrics=metrics)

    assert read_rank_metrics(paths) == (metrics,)


def test_read_lineage_manifest_missing_file_returns_empty_manifest(
    tmp_path: Path,
) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    assert read_lineage_manifest(paths) == {
        "lineage_id": "lineage-a",
        "artifact_schema_version": 1,
        "ranks": {},
    }


def test_write_and_read_lineage_manifest_round_trip(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    manifest = {
        "lineage_id": "lineage-a",
        "artifact_schema_version": 1,
        "ranks": {},
    }

    written_path = write_lineage_manifest(paths, manifest)

    assert written_path == paths.manifest_path
    assert read_lineage_manifest(paths) == manifest
    assert read_lineage_manifest(paths.lineage_dir) == manifest
    assert paths.manifest_path.read_text().endswith("\n")


def test_record_rank_manifest_entry_records_rank_1_artifacts(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    manifest = record_rank_manifest_entry(paths=paths, status="accepted")

    assert manifest["ranks"]["1"] == {
        "status": "accepted",
        "checkpoint": "rank_1/checkpoint_latest.pt",
        "config": "rank_1/config.json",
        "metrics": "rank_1/metrics.jsonl",
    }
    assert read_lineage_manifest(paths) == manifest


def test_record_rank_manifest_entry_records_rank_2_parent(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )

    manifest = record_rank_manifest_entry(
        paths=paths,
        status="running",
        parent_checkpoint="rank_1/checkpoint_latest.pt",
    )

    assert manifest["ranks"]["2"] == {
        "status": "running",
        "checkpoint": "rank_2/checkpoint_latest.pt",
        "config": "rank_2/config.json",
        "metrics": "rank_2/metrics.jsonl",
        "parent_rank": 1,
        "parent_checkpoint": "rank_1/checkpoint_latest.pt",
    }


def test_find_accepted_parent_checkpoint_returns_lineage_path(tmp_path: Path) -> None:
    rank_1_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    rank_2_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    record_rank_manifest_entry(paths=rank_1_paths, status="accepted")

    assert (
        find_accepted_parent_checkpoint(rank_2_paths)
        == rank_2_paths.lineage_dir / "rank_1" / "checkpoint_latest.pt"
    )


def test_find_accepted_parent_checkpoint_rejects_rank_1(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    with pytest.raises(ValueError, match="rank 1 has no parent checkpoint"):
        find_accepted_parent_checkpoint(paths)


def test_find_accepted_parent_checkpoint_rejects_missing_parent(
    tmp_path: Path,
) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )

    with pytest.raises(ValueError, match="accepted parent checkpoint is missing"):
        find_accepted_parent_checkpoint(paths)


def test_find_accepted_parent_checkpoint_rejects_non_accepted_parent(
    tmp_path: Path,
) -> None:
    rank_1_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    rank_2_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    record_rank_manifest_entry(paths=rank_1_paths, status="running")

    with pytest.raises(ValueError, match="parent rank is not accepted"):
        find_accepted_parent_checkpoint(rank_2_paths)


def test_record_rank_manifest_entry_rejects_unknown_status(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    with pytest.raises(ValueError, match="recognized manifest status"):
        record_rank_manifest_entry(paths=paths, status="done")


def test_manifest_status_set_matches_artifact_contract() -> None:
    assert MANIFEST_STATUSES == {
        "not_started",
        "running",
        "accepted",
        "failed",
        "superseded",
    }


def test_build_checkpoint_payload_records_rank_1_required_fields() -> None:
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=10,
    )

    payload = build_checkpoint_payload(
        config=config,
        episode_index=3,
        stats={"episode_return": 1.5},
        policy_state_dict={"weight": [1.0, 2.0]},
        optimizer_state_dict={"step": 3},
    )

    assert payload["rank"] == 1
    assert payload["lineage_id"] == "lineage-a"
    assert payload["episode_index"] == 3
    assert payload["config"] == config.to_json_dict()
    assert payload["stats"] == {"episode_return": 1.5}
    assert payload["policy_state_dict"] == {"weight": [1.0, 2.0]}
    assert payload["optimizer_state_dict"] == {"step": 3}
    assert payload["artifact_schema_version"] == config.artifact_schema_version


def test_save_and_load_latest_checkpoint_round_trip(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=10,
    )
    payload = build_checkpoint_payload(
        config=config,
        episode_index=0,
        stats={"episode_return": 0.0},
        policy_state_dict={"bias": [0.25]},
        optimizer_state_dict={"step": 0},
    )

    written_path = save_latest_checkpoint(paths=paths, payload=payload)

    assert written_path == paths.checkpoint_latest_path
    assert paths.checkpoint_latest_path.exists()
    assert load_latest_checkpoint(paths) == payload
    assert load_latest_checkpoint(paths.checkpoint_latest_path) == payload


def test_save_latest_checkpoint_rejects_rank_mismatch(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    payload = {
        "rank": 1,
        "lineage_id": "lineage-a",
        "artifact_schema_version": 1,
    }

    with pytest.raises(ValueError, match="checkpoint rank must match"):
        save_latest_checkpoint(paths=paths, payload=payload)


def test_save_latest_checkpoint_rejects_lineage_mismatch(tmp_path: Path) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    payload = {
        "rank": 1,
        "lineage_id": "lineage-b",
        "artifact_schema_version": 1,
    }

    with pytest.raises(ValueError, match="checkpoint lineage_id must match"):
        save_latest_checkpoint(paths=paths, payload=payload)


def test_build_rank_2_checkpoint_payload_records_parent_metadata() -> None:
    config = TowerRankConfig(
        rank=2,
        lineage_id="lineage-a",
        episode_budget=10,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
    )

    payload = build_checkpoint_payload(
        config=config,
        episode_index=4,
        stats={"episode_return": 2.0},
        policy_state_dict={"child_weight": [1.0]},
        optimizer_state_dict={"step": 4},
        parent_checkpoint_id="parent-001",
        parent_config_hash="abc123",
        parent_artifact_schema_version=1,
    )

    assert payload["parent_rank"] == 1
    assert payload["parent_checkpoint"] == "rank_1/checkpoint_latest.pt"
    assert payload["parent_checkpoint_id"] == "parent-001"
    assert payload["parent_config_hash"] == "abc123"
    assert payload["parent_artifact_schema_version"] == 1
    assert "parent_policy_state_dict" not in payload


def test_build_rank_2_checkpoint_payload_requires_parent_checkpoint() -> None:
    config = TowerRankConfig(
        rank=2,
        lineage_id="lineage-a",
        episode_budget=10,
    )

    with pytest.raises(ValueError, match="requires parent_checkpoint"):
        build_checkpoint_payload(
            config=config,
            episode_index=0,
            stats={},
            policy_state_dict={},
            optimizer_state_dict={},
        )


def test_build_checkpoint_payload_rejects_invalid_episode_index() -> None:
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=10,
    )

    with pytest.raises(ValueError, match="episode_index must be non-negative"):
        build_checkpoint_payload(
            config=config,
            episode_index=-1,
            stats={},
            policy_state_dict={},
            optimizer_state_dict={},
        )


def test_complete_rank_1_artifact_flow_writes_expected_files(
    tmp_path: Path,
) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=2,
    )
    payload = build_checkpoint_payload(
        config=config,
        episode_index=1,
        stats={"episode_return": 3.0},
        policy_state_dict={"weight": [1.0]},
        optimizer_state_dict={"step": 1},
    )

    write_rank_config(config=config, paths=paths)
    append_rank_metrics(
        paths=paths,
        metrics={
            "rank": 1,
            "episode_index": 1,
            "episode_return": 3.0,
            "terminal_success": True,
        },
    )
    save_latest_checkpoint(paths=paths, payload=payload)
    manifest = record_rank_manifest_entry(paths=paths, status="accepted")

    assert paths.config_path.exists()
    assert paths.metrics_path.exists()
    assert paths.checkpoint_latest_path.exists()
    assert paths.manifest_path.exists()
    assert read_rank_config(paths) == config
    assert read_rank_metrics(paths) == (
        {
            "episode_index": 1,
            "episode_return": 3.0,
            "rank": 1,
            "terminal_success": True,
        },
    )
    assert load_latest_checkpoint(paths) == payload
    assert manifest["ranks"]["1"]["status"] == "accepted"


def test_complete_rank_2_artifact_flow_records_parent_consistently(
    tmp_path: Path,
) -> None:
    rank_1_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    rank_2_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    rank_1_config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=2,
    )
    rank_2_config = TowerRankConfig(
        rank=2,
        lineage_id="lineage-a",
        episode_budget=2,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
    )

    save_latest_checkpoint(
        paths=rank_1_paths,
        payload=build_checkpoint_payload(
            config=rank_1_config,
            episode_index=1,
            stats={"episode_return": 1.0},
            policy_state_dict={"parent_weight": [1.0]},
            optimizer_state_dict={"step": 1},
        ),
    )
    record_rank_manifest_entry(paths=rank_1_paths, status="accepted")
    write_rank_config(config=rank_2_config, paths=rank_2_paths)
    rank_2_payload = build_checkpoint_payload(
        config=rank_2_config,
        episode_index=1,
        stats={"episode_return": 2.0},
        policy_state_dict={"child_weight": [2.0]},
        optimizer_state_dict={"step": 1},
    )
    save_latest_checkpoint(paths=rank_2_paths, payload=rank_2_payload)
    manifest = record_rank_manifest_entry(
        paths=rank_2_paths,
        status="accepted",
        parent_checkpoint=rank_2_config.parent_checkpoint,
    )

    assert find_accepted_parent_checkpoint(rank_2_paths) == rank_1_paths.checkpoint_latest_path
    assert load_accepted_parent_checkpoint(rank_2_paths)["rank"] == 1
    assert read_rank_config(rank_2_paths).parent_checkpoint == "rank_1/checkpoint_latest.pt"
    assert load_latest_checkpoint(rank_2_paths)["parent_checkpoint"] == (
        "rank_1/checkpoint_latest.pt"
    )
    assert manifest["ranks"]["2"]["parent_rank"] == 1
    assert manifest["ranks"]["2"]["parent_checkpoint"] == "rank_1/checkpoint_latest.pt"
    assert "parent_policy_state_dict" not in load_latest_checkpoint(rank_2_paths)


def test_checkpoint_module_does_not_import_policy_or_training_loop_modules() -> None:
    project_root = Path(__file__).parents[3]
    source_text = (project_root / "tower" / "train" / "checkpoint.py").read_text()

    forbidden_imports = (
        "from tower.policy",
        "import tower.policy",
        "from tower.models",
        "import tower.models",
        "from tower.train.rollout",
        "import tower.train.rollout",
    )

    assert not any(forbidden in source_text for forbidden in forbidden_imports)


def test_load_accepted_parent_checkpoint_returns_parent_payload(
    tmp_path: Path,
) -> None:
    rank_1_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    rank_2_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=1,
    )
    parent_payload = build_checkpoint_payload(
        config=config,
        episode_index=0,
        stats={"episode_return": 1.0},
        policy_state_dict={"parent_weight": [1.0]},
        optimizer_state_dict={"step": 1},
    )
    save_latest_checkpoint(paths=rank_1_paths, payload=parent_payload)
    record_rank_manifest_entry(paths=rank_1_paths, status="accepted")

    loaded = load_accepted_parent_checkpoint(rank_2_paths)

    assert loaded == parent_payload
    assert loaded["policy_state_dict"] == {"parent_weight": [1.0]}


def test_load_accepted_parent_checkpoint_rejects_rank_1(
    tmp_path: Path,
) -> None:
    paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )

    with pytest.raises(ValueError, match="rank 1 has no parent checkpoint"):
        load_accepted_parent_checkpoint(paths)


def test_load_accepted_parent_checkpoint_rejects_parent_rank_mismatch(
    tmp_path: Path,
) -> None:
    rank_1_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    rank_2_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    wrong_parent_payload = {
        "rank": 2,
        "lineage_id": "lineage-a",
        "artifact_schema_version": 1,
    }
    rank_1_paths.rank_dir.mkdir(parents=True)
    torch.save(wrong_parent_payload, rank_1_paths.checkpoint_latest_path)
    record_rank_manifest_entry(paths=rank_1_paths, status="accepted")

    with pytest.raises(ValueError, match="parent checkpoint rank does not match"):
        load_accepted_parent_checkpoint(rank_2_paths)


def test_load_accepted_parent_checkpoint_rejects_parent_lineage_mismatch(
    tmp_path: Path,
) -> None:
    rank_1_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=1,
        artifact_root=tmp_path,
    )
    rank_2_paths = TowerArtifactPaths(
        lineage_id="lineage-a",
        rank=2,
        artifact_root=tmp_path,
    )
    wrong_parent_payload = {
        "rank": 1,
        "lineage_id": "lineage-b",
        "artifact_schema_version": 1,
    }
    # Write directly so this test can isolate load-time lineage validation.
    rank_1_paths.rank_dir.mkdir(parents=True)
    torch.save(wrong_parent_payload, rank_1_paths.checkpoint_latest_path)
    record_rank_manifest_entry(paths=rank_1_paths, status="accepted")

    with pytest.raises(ValueError, match="parent checkpoint lineage_id does not match"):
        load_accepted_parent_checkpoint(rank_2_paths)
