"""Trajectory diagnostics serialization for tower training artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch

from tower.train.config import ARTIFACT_SCHEMA_VERSION
from tower.train.trajectory import TowerTrajectory

EpisodeKind = Literal["training", "final_inference"]


def reward_diagnostics_rows(
    *,
    trajectory: TowerTrajectory,
    lineage_id: str,
    episode_index: int,
    episode_kind: EpisodeKind,
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION,
) -> tuple[dict[str, object], ...]:
    """Return JSON-compatible per-step reward diagnostics rows."""
    if not isinstance(trajectory, TowerTrajectory):
        raise TypeError("trajectory must be a TowerTrajectory")
    if not isinstance(lineage_id, str):
        raise TypeError("lineage_id must be a string")
    if not lineage_id:
        raise ValueError("lineage_id must not be empty")
    if not isinstance(episode_index, int):
        raise TypeError("episode_index must be an int")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if episode_kind not in {"training", "final_inference"}:
        raise ValueError("episode_kind must be training or final_inference")
    if not isinstance(artifact_schema_version, int):
        raise TypeError("artifact_schema_version must be an int")
    if artifact_schema_version < 1:
        raise ValueError("artifact_schema_version must be at least 1")

    rows = []
    for step in trajectory.steps:
        row: dict[str, object] = {
            "artifact_schema_version": artifact_schema_version,
            "lineage_id": lineage_id,
            "rank": step.rank,
            "episode_index": episode_index,
            "episode_kind": episode_kind,
            "step_index": step.step_index,
            "source_state": list(step.source_state),
            "assembled_action": list(step.assembled_action),
            "attempted_target_state": list(step.attempted_target_state),
            "realized_next_state": list(step.realized_next_state),
            "reward": float(step.reward.reward),
            "hard_violation": step.reward.hard_violation,
            "is_terminal_success": step.reward.is_terminal_success,
            "reward_diagnostics": to_json_compatible(
                step.reward.diagnostics,
                field_name="reward_diagnostics",
            ),
            "terminated": step.terminated,
            "truncated": step.truncated,
            "outcome": step.outcome,
        }
        if step.parent_state is not None:
            row["parent_state"] = list(step.parent_state)
        if step.parent_action is not None:
            row["parent_action"] = list(step.parent_action)
        if step.active_choice is not None:
            row["active_choice"] = step.active_choice
        if step.diagnostics:
            row["step_diagnostics"] = to_json_compatible(
                step.diagnostics,
                field_name="step_diagnostics",
            )
        rows.append(row)

    return tuple(rows)


def to_json_compatible(value: object, *, field_name: str) -> object:
    """Return a JSON-compatible copy of a nested diagnostics value."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must not contain torch.Tensor values")
    if isinstance(value, tuple | list):
        return [
            to_json_compatible(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            converted[key] = to_json_compatible(
                item,
                field_name=f"{field_name}.{key}",
            )
        return converted
    raise TypeError(f"{field_name} must be JSON-compatible")
