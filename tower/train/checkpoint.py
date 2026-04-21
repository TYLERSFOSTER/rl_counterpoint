"""Artifact path contracts for tower training checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import torch

from tower.state_action import validate_rank
from tower.train.config import (
    ARTIFACT_SCHEMA_VERSION,
    TowerRankConfig,
    _validate_json_mapping,
)

DEFAULT_TOWER_ARTIFACT_ROOT = Path("artifacts") / "tower"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.jsonl"
REWARD_DIAGNOSTICS_FILENAME = "reward_diagnostics.jsonl"
CHECKPOINT_LATEST_FILENAME = "checkpoint_latest.pt"
EXAMPLE_EPISODE_FILENAME = "example_episode.mid"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_STATUSES = frozenset(
    {
        "not_started",
        "running",
        "accepted",
        "failed",
        "superseded",
    }
)


@dataclass(frozen=True)
class TowerArtifactPaths:
    """Deterministic artifact paths for one tower lineage rank."""

    lineage_id: str
    rank: int
    artifact_root: Path = DEFAULT_TOWER_ARTIFACT_ROOT

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        if not isinstance(self.lineage_id, str):
            raise TypeError("lineage_id must be a string")
        if not self.lineage_id:
            raise ValueError("lineage_id must not be empty")
        if not isinstance(self.artifact_root, Path):
            raise TypeError("artifact_root must be a Path")

    @property
    def lineage_dir(self) -> Path:
        """Return artifacts/tower/<lineage_id>."""
        return self.artifact_root / self.lineage_id

    @property
    def rank_dir(self) -> Path:
        """Return artifacts/tower/<lineage_id>/rank_<k>."""
        return self.lineage_dir / f"rank_{self.rank}"

    @property
    def manifest_path(self) -> Path:
        """Return the lineage manifest path."""
        return self.lineage_dir / MANIFEST_FILENAME

    @property
    def config_path(self) -> Path:
        """Return the rank config path."""
        return self.rank_dir / CONFIG_FILENAME

    @property
    def metrics_path(self) -> Path:
        """Return the rank metrics JSONL path."""
        return self.rank_dir / METRICS_FILENAME

    @property
    def reward_diagnostics_path(self) -> Path:
        """Return the rank reward diagnostics JSONL path."""
        return self.rank_dir / REWARD_DIAGNOSTICS_FILENAME

    @property
    def checkpoint_latest_path(self) -> Path:
        """Return the rolling latest checkpoint path."""
        return self.rank_dir / CHECKPOINT_LATEST_FILENAME

    @property
    def example_episode_path(self) -> Path:
        """Return the recommended example episode MIDI path."""
        return self.rank_dir / EXAMPLE_EPISODE_FILENAME

    @property
    def relative_config_path(self) -> Path:
        """Return the rank config path relative to the lineage directory."""
        return Path(f"rank_{self.rank}") / CONFIG_FILENAME

    @property
    def relative_metrics_path(self) -> Path:
        """Return the metrics path relative to the lineage directory."""
        return Path(f"rank_{self.rank}") / METRICS_FILENAME

    @property
    def relative_reward_diagnostics_path(self) -> Path:
        """Return the reward diagnostics path relative to the lineage directory."""
        return Path(f"rank_{self.rank}") / REWARD_DIAGNOSTICS_FILENAME

    @property
    def relative_checkpoint_latest_path(self) -> Path:
        """Return the checkpoint path relative to the lineage directory."""
        return Path(f"rank_{self.rank}") / CHECKPOINT_LATEST_FILENAME

    @property
    def relative_example_episode_path(self) -> Path:
        """Return the example MIDI path relative to the lineage directory."""
        return Path(f"rank_{self.rank}") / EXAMPLE_EPISODE_FILENAME


def write_rank_config(
    *,
    config: TowerRankConfig,
    paths: TowerArtifactPaths,
) -> Path:
    """Persist one rank config as deterministic JSON."""
    _validate_config_matches_paths(config=config, paths=paths)
    paths.rank_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(
        json.dumps(config.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths.config_path


def read_rank_config(path_or_paths: Path | TowerArtifactPaths) -> TowerRankConfig:
    """Read one rank config from a config path or artifact path bundle."""
    path = (
        path_or_paths.config_path
        if isinstance(path_or_paths, TowerArtifactPaths)
        else path_or_paths
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("config JSON must contain an object")
    return TowerRankConfig.from_json_dict(payload)


def append_rank_metrics(
    *,
    paths: TowerArtifactPaths,
    metrics: Mapping[str, object],
) -> Path:
    """Append one JSON metrics row for a rank episode."""
    _validate_metrics(metrics=metrics, paths=paths)
    paths.rank_dir.mkdir(parents=True, exist_ok=True)
    with paths.metrics_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(dict(metrics), sort_keys=True) + "\n")
    return paths.metrics_path


def read_rank_metrics(
    path_or_paths: Path | TowerArtifactPaths,
) -> tuple[dict[str, object], ...]:
    """Read rank metrics JSONL rows in append order."""
    path = (
        path_or_paths.metrics_path
        if isinstance(path_or_paths, TowerArtifactPaths)
        else path_or_paths
    )
    if not path.exists():
        return ()

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("metrics JSONL rows must contain objects")
        _validate_json_mapping(payload, field_name="metrics")
        rows.append(payload)
    return tuple(rows)


def append_reward_diagnostics(
    *,
    paths: TowerArtifactPaths,
    rows: tuple[Mapping[str, object], ...],
) -> Path:
    """Append JSON reward diagnostics rows for one rank."""
    if not isinstance(rows, tuple):
        raise TypeError("rows must be a tuple")
    paths.rank_dir.mkdir(parents=True, exist_ok=True)
    with paths.reward_diagnostics_path.open(
        "a",
        encoding="utf-8",
    ) as diagnostics_file:
        for row in rows:
            _validate_reward_diagnostics(row=row, paths=paths)
            diagnostics_file.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return paths.reward_diagnostics_path


def read_reward_diagnostics(
    path_or_paths: Path | TowerArtifactPaths,
) -> tuple[dict[str, object], ...]:
    """Read reward diagnostics JSONL rows in append order."""
    path = (
        path_or_paths.reward_diagnostics_path
        if isinstance(path_or_paths, TowerArtifactPaths)
        else path_or_paths
    )
    if not path.exists():
        return ()

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("reward diagnostics JSONL rows must contain objects")
        _validate_json_mapping(payload, field_name="reward_diagnostics")
        rows.append(payload)
    return tuple(rows)


def read_lineage_manifest(
    path_or_paths: Path | TowerArtifactPaths,
) -> dict[str, object]:
    """Read a lineage manifest, or return an empty manifest if missing."""
    lineage_dir, manifest_path, lineage_id = _lineage_manifest_locations(path_or_paths)
    if not manifest_path.exists():
        return {
            "lineage_id": lineage_id,
            "artifact_schema_version": 1,
            "ranks": {},
        }

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("manifest JSON must contain an object")
    _validate_manifest_payload(payload)
    if payload["lineage_id"] != lineage_id:
        raise ValueError("manifest lineage_id does not match lineage directory")
    # lineage_dir is intentionally computed here so direct Path calls validate shape.
    _ = lineage_dir
    return payload


def write_lineage_manifest(
    path_or_paths: Path | TowerArtifactPaths,
    manifest: Mapping[str, object],
) -> Path:
    """Persist a lineage manifest as deterministic JSON."""
    _, manifest_path, lineage_id = _lineage_manifest_locations(path_or_paths)
    _validate_manifest_payload(manifest)
    if manifest["lineage_id"] != lineage_id:
        raise ValueError("manifest lineage_id does not match lineage directory")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def record_rank_manifest_entry(
    *,
    paths: TowerArtifactPaths,
    status: str,
    parent_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Update the lineage manifest entry for one rank."""
    if status not in MANIFEST_STATUSES:
        raise ValueError("status is not a recognized manifest status")

    manifest = read_lineage_manifest(paths)
    ranks = manifest["ranks"]
    if not isinstance(ranks, dict):
        raise TypeError("manifest ranks must be an object")

    entry: dict[str, object] = {
        "status": status,
        "checkpoint": paths.relative_checkpoint_latest_path.as_posix(),
        "config": paths.relative_config_path.as_posix(),
        "metrics": paths.relative_metrics_path.as_posix(),
        "reward_diagnostics": paths.relative_reward_diagnostics_path.as_posix(),
    }

    if paths.rank > 1:
        if parent_checkpoint is None:
            parent_checkpoint = (
                Path(f"rank_{paths.rank - 1}") / CHECKPOINT_LATEST_FILENAME
            )
        entry["parent_rank"] = paths.rank - 1
        entry["parent_checkpoint"] = Path(parent_checkpoint).as_posix()

    ranks[str(paths.rank)] = entry
    write_lineage_manifest(paths, manifest)
    return manifest


def find_accepted_parent_checkpoint(paths: TowerArtifactPaths) -> Path:
    """Return the accepted parent checkpoint path for a child rank."""
    if paths.rank <= 1:
        raise ValueError("rank 1 has no parent checkpoint")

    manifest = read_lineage_manifest(paths)
    ranks = manifest["ranks"]
    if not isinstance(ranks, dict):
        raise TypeError("manifest ranks must be an object")

    parent_rank = paths.rank - 1
    parent_entry = ranks.get(str(parent_rank))
    if not isinstance(parent_entry, dict):
        raise ValueError("accepted parent checkpoint is missing from manifest")
    if parent_entry.get("status") != "accepted":
        raise ValueError("parent rank is not accepted")

    parent_checkpoint = parent_entry.get("checkpoint")
    if not isinstance(parent_checkpoint, str):
        raise ValueError("parent checkpoint path is missing from manifest")

    return paths.lineage_dir / parent_checkpoint


def build_checkpoint_payload(
    *,
    config: TowerRankConfig,
    episode_index: int,
    stats: Mapping[str, object],
    policy_state_dict: Mapping[str, object],
    optimizer_state_dict: Mapping[str, object],
    parent_checkpoint: str | Path | None = None,
    parent_checkpoint_id: str | None = None,
    parent_config_hash: str | None = None,
    parent_artifact_schema_version: int | None = None,
) -> dict[str, object]:
    """Build a JSON-like rolling latest checkpoint payload shell."""
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    _validate_json_mapping(stats, field_name="stats")
    _validate_mapping(policy_state_dict, field_name="policy_state_dict")
    _validate_mapping(optimizer_state_dict, field_name="optimizer_state_dict")

    payload: dict[str, object] = {
        "rank": config.rank,
        "lineage_id": config.lineage_id,
        "episode_index": episode_index,
        "config": config.to_json_dict(),
        "stats": dict(stats),
        "policy_state_dict": dict(policy_state_dict),
        "optimizer_state_dict": dict(optimizer_state_dict),
        "artifact_schema_version": config.artifact_schema_version,
    }

    if config.rank > 1:
        checkpoint = parent_checkpoint if parent_checkpoint is not None else config.parent_checkpoint
        if checkpoint is None:
            raise ValueError("rank greater than 1 checkpoint requires parent_checkpoint")
        payload["parent_rank"] = config.rank - 1
        payload["parent_checkpoint"] = Path(checkpoint).as_posix()
        if parent_checkpoint_id is not None:
            payload["parent_checkpoint_id"] = parent_checkpoint_id
        if parent_config_hash is not None:
            payload["parent_config_hash"] = parent_config_hash
        if parent_artifact_schema_version is not None:
            payload["parent_artifact_schema_version"] = parent_artifact_schema_version

    return payload


def save_latest_checkpoint(
    *,
    paths: TowerArtifactPaths,
    payload: Mapping[str, object],
) -> Path:
    """Save the rolling latest checkpoint payload for one rank."""
    _validate_checkpoint_payload_matches_paths(payload=payload, paths=paths)
    paths.rank_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), paths.checkpoint_latest_path)
    return paths.checkpoint_latest_path


def load_latest_checkpoint(path_or_paths: Path | TowerArtifactPaths) -> dict[str, object]:
    """Load a rolling latest checkpoint payload."""
    path = (
        path_or_paths.checkpoint_latest_path
        if isinstance(path_or_paths, TowerArtifactPaths)
        else path_or_paths
    )
    payload = torch.load(path, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dict")
    return payload


def load_accepted_parent_checkpoint(paths: TowerArtifactPaths) -> dict[str, object]:
    """Load the accepted parent checkpoint for a child rank."""
    parent_checkpoint_path = find_accepted_parent_checkpoint(paths)
    payload = load_latest_checkpoint(parent_checkpoint_path)
    expected_parent_rank = paths.rank - 1
    if payload.get("rank") != expected_parent_rank:
        raise ValueError("parent checkpoint rank does not match child rank")
    if payload.get("lineage_id") != paths.lineage_id:
        raise ValueError("parent checkpoint lineage_id does not match child lineage")
    return payload


def _validate_config_matches_paths(
    *,
    config: TowerRankConfig,
    paths: TowerArtifactPaths,
) -> None:
    if config.rank != paths.rank:
        raise ValueError("config rank must match artifact paths rank")
    if config.lineage_id != paths.lineage_id:
        raise ValueError("config lineage_id must match artifact paths lineage_id")


def _validate_metrics(
    *,
    metrics: Mapping[str, object],
    paths: TowerArtifactPaths,
) -> None:
    _validate_json_mapping(metrics, field_name="metrics")
    if "rank" in metrics and metrics["rank"] != paths.rank:
        raise ValueError("metrics rank must match artifact paths rank")
    if "episode_index" in metrics and not isinstance(metrics["episode_index"], int):
        raise TypeError("metrics episode_index must be an int")


def _validate_reward_diagnostics(
    *,
    row: Mapping[str, object],
    paths: TowerArtifactPaths,
) -> None:
    _validate_json_mapping(row, field_name="reward_diagnostics")
    if row.get("rank") != paths.rank:
        raise ValueError("reward diagnostics rank must match artifact paths rank")
    if row.get("lineage_id") != paths.lineage_id:
        raise ValueError(
            "reward diagnostics lineage_id must match artifact paths lineage_id"
        )
    if not isinstance(row.get("episode_index"), int):
        raise TypeError("reward diagnostics episode_index must be an int")
    if not isinstance(row.get("step_index"), int):
        raise TypeError("reward diagnostics step_index must be an int")


def _validate_checkpoint_payload_matches_paths(
    *,
    payload: Mapping[str, object],
    paths: TowerArtifactPaths,
) -> None:
    rank = payload.get("rank")
    lineage_id = payload.get("lineage_id")
    if rank != paths.rank:
        raise ValueError("checkpoint rank must match artifact paths rank")
    if lineage_id != paths.lineage_id:
        raise ValueError("checkpoint lineage_id must match artifact paths lineage_id")
    schema_version = payload.get("artifact_schema_version")
    if not isinstance(schema_version, int) or schema_version < ARTIFACT_SCHEMA_VERSION:
        raise ValueError("checkpoint artifact_schema_version is invalid")


def _validate_mapping(payload: Mapping[str, object], *, field_name: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must be a mapping")


def _lineage_manifest_locations(
    path_or_paths: Path | TowerArtifactPaths,
) -> tuple[Path, Path, str]:
    if isinstance(path_or_paths, TowerArtifactPaths):
        return (
            path_or_paths.lineage_dir,
            path_or_paths.manifest_path,
            path_or_paths.lineage_id,
        )

    lineage_dir = path_or_paths
    if not isinstance(lineage_dir, Path):
        raise TypeError("path_or_paths must be a Path or TowerArtifactPaths")
    return lineage_dir, lineage_dir / MANIFEST_FILENAME, lineage_dir.name


def _validate_manifest_payload(manifest: Mapping[str, object]) -> None:
    _validate_json_mapping(manifest, field_name="manifest")
    lineage_id = manifest.get("lineage_id")
    if not isinstance(lineage_id, str) or not lineage_id:
        raise ValueError("manifest lineage_id must be a non-empty string")
    schema_version = manifest.get("artifact_schema_version")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("manifest artifact_schema_version must be at least 1")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, dict):
        raise TypeError("manifest ranks must be an object")

    for rank_key, entry in ranks.items():
        if not isinstance(rank_key, str):
            raise TypeError("manifest rank keys must be strings")
        if not isinstance(entry, dict):
            raise TypeError("manifest rank entries must be objects")
        status = entry.get("status")
        if status not in MANIFEST_STATUSES:
            raise ValueError("manifest rank status is not recognized")
