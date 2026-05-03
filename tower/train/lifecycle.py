"""Lifecycle marker files for long-running tower training commands."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Mapping

from tower.train.checkpoint import TowerArtifactPaths
from tower.train.config import _validate_json_mapping

HEARTBEAT_FILENAME = "heartbeat.json"
COMPLETION_FILENAME = "completion.json"
FAILURE_FILENAME = "failure.json"


def heartbeat_path(path_or_paths: Path | TowerArtifactPaths) -> Path:
    """Return the lineage heartbeat path."""
    return _lineage_dir(path_or_paths) / HEARTBEAT_FILENAME


def completion_path(path_or_paths: Path | TowerArtifactPaths) -> Path:
    """Return the lineage completion path."""
    return _lineage_dir(path_or_paths) / COMPLETION_FILENAME


def failure_path(path_or_paths: Path | TowerArtifactPaths) -> Path:
    """Return the lineage failure path."""
    return _lineage_dir(path_or_paths) / FAILURE_FILENAME


def write_run_heartbeat(
    path_or_paths: Path | TowerArtifactPaths,
    *,
    lineage_id: str,
    stage: str,
    status: str = "running",
    rank: int | None = None,
    completed_episodes: int | None = None,
    episode_budget: int | None = None,
    metrics: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write or overwrite the lineage heartbeat marker."""
    payload: dict[str, object] = {
        "lineage_id": lineage_id,
        "stage": stage,
        "status": status,
        "updated_at": _utc_now_iso(),
    }
    if rank is not None:
        payload["rank"] = rank
    if completed_episodes is not None:
        payload["completed_episodes"] = completed_episodes
    if episode_budget is not None:
        payload["episode_budget"] = episode_budget
    if metrics is not None:
        _validate_json_mapping(metrics, field_name="heartbeat.metrics")
        payload["metrics"] = dict(metrics)
    if extra is not None:
        _validate_json_mapping(extra, field_name="heartbeat.extra")
        payload["extra"] = dict(extra)
    return _write_json(heartbeat_path(path_or_paths), payload)


def write_run_completion(
    path_or_paths: Path | TowerArtifactPaths,
    *,
    lineage_id: str,
    stage: str,
    summary: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write the lineage completion marker."""
    payload: dict[str, object] = {
        "lineage_id": lineage_id,
        "stage": stage,
        "status": "completed",
        "completed_at": _utc_now_iso(),
    }
    if summary is not None:
        _validate_json_mapping(summary, field_name="completion.summary")
        payload["summary"] = dict(summary)
    if extra is not None:
        _validate_json_mapping(extra, field_name="completion.extra")
        payload["extra"] = dict(extra)
    return _write_json(completion_path(path_or_paths), payload)


def write_run_failure(
    path_or_paths: Path | TowerArtifactPaths,
    *,
    lineage_id: str,
    stage: str,
    error_type: str,
    error_message: str,
    exception_log: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write the lineage failure marker."""
    payload: dict[str, object] = {
        "lineage_id": lineage_id,
        "stage": stage,
        "status": "failed",
        "failed_at": _utc_now_iso(),
        "error_type": error_type,
        "error_message": error_message,
    }
    if exception_log is not None:
        payload["exception_log"] = exception_log
    if extra is not None:
        _validate_json_mapping(extra, field_name="failure.extra")
        payload["extra"] = dict(extra)
    return _write_json(failure_path(path_or_paths), payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    _validate_json_mapping(payload, field_name=path.name.removesuffix(".json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _lineage_dir(path_or_paths: Path | TowerArtifactPaths) -> Path:
    if isinstance(path_or_paths, TowerArtifactPaths):
        return path_or_paths.lineage_dir
    if not isinstance(path_or_paths, Path):
        raise TypeError("path_or_paths must be a Path or TowerArtifactPaths")
    return path_or_paths


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
