"""Tests for induced lower-rank graph artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from tower.graph.induced import (
    build_induced_rank1_graph_payload,
    build_induced_rank2_graph_payload,
    induced_rank1_graph_artifact_path,
    induced_rank1_graph_cache_key,
    induced_rank2_graph_artifact_path,
    induced_rank2_graph_cache_key,
    write_induced_rank1_graph_artifact,
    write_induced_rank2_graph_artifact,
)
from tower.graph.spec import TowerGraphSpec


def test_induced_rank1_payload_contains_only_projected_valid_rank2_nodes() -> None:
    payload = build_induced_rank1_graph_payload(
        source_spec=TowerGraphSpec(rank=2, pitch_min=63, pitch_max=67, max_step_size=1)
    )

    assert payload["source_rank"] == 2
    assert payload["target_rank"] == 1
    assert payload["node_image"] == [[64]]


def test_induced_rank1_payload_contains_projected_valid_edges() -> None:
    payload = build_induced_rank1_graph_payload(
        source_spec=TowerGraphSpec(rank=2, pitch_min=63, pitch_max=67, max_step_size=1)
    )

    assert payload["edge_image"] == []


def test_induced_rank1_cache_key_depends_on_source_graph_spec() -> None:
    left = induced_rank1_graph_cache_key(
        source_spec=TowerGraphSpec(rank=2, pitch_min=36, pitch_max=84, max_step_size=7)
    )
    right = induced_rank1_graph_cache_key(
        source_spec=TowerGraphSpec(rank=2, pitch_min=36, pitch_max=84, max_step_size=1)
    )

    assert left != right


def test_induced_rank1_cache_key_depends_on_key_pitch_class() -> None:
    left = induced_rank1_graph_cache_key(
        source_spec=TowerGraphSpec(
            rank=2,
            key_pitch_class=0,
            pitch_min=36,
            pitch_max=84,
            max_step_size=7,
        )
    )
    right = induced_rank1_graph_cache_key(
        source_spec=TowerGraphSpec(
            rank=2,
            key_pitch_class=5,
            pitch_min=36,
            pitch_max=84,
            max_step_size=7,
        )
    )

    assert left != right


def test_induced_rank1_artifact_path_is_deterministic() -> None:
    spec = TowerGraphSpec(rank=2, pitch_min=36, pitch_max=84, max_step_size=7)

    left = induced_rank1_graph_artifact_path(
        source_spec=spec,
        artifact_root=Path("/tmp/artifacts-a"),
    )
    right = induced_rank1_graph_artifact_path(
        source_spec=spec,
        artifact_root=Path("/tmp/artifacts-a"),
    )

    assert left == right
    assert left.name.endswith(".json")


def test_write_induced_rank1_graph_artifact_persists_payload(tmp_path: Path) -> None:
    spec = TowerGraphSpec(rank=2, pitch_min=63, pitch_max=67, max_step_size=1)

    artifact_path = write_induced_rank1_graph_artifact(
        source_spec=spec,
        artifact_root=tmp_path,
    )

    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "induced_rank1_graph"
    assert payload["artifact_schema_version"] == 2
    assert payload["rank2_legality_contract_version"] == "tower.rank2_legality.v4"
    assert payload["node_image"] == [[64]]


def test_induced_rank2_payload_contains_only_projected_valid_rank3_nodes() -> None:
    payload = build_induced_rank2_graph_payload(
        source_spec=TowerGraphSpec(
            rank=3,
            key_pitch_class=0,
            pitch_min=60,
            pitch_max=69,
            max_step_size=2,
        )
    )

    assert payload["source_rank"] == 3
    assert payload["target_rank"] == 2
    assert payload["node_image"] == [[60, 67], [60, 68], [62, 69]]


def test_induced_rank2_payload_contains_projected_valid_edges() -> None:
    payload = build_induced_rank2_graph_payload(
        source_spec=TowerGraphSpec(
            rank=3,
            key_pitch_class=0,
            pitch_min=60,
            pitch_max=69,
            max_step_size=2,
        )
    )

    assert payload["edge_image"] == [
        {"source": [60, 68], "target": [62, 69]},
        {"source": [62, 69], "target": [60, 68]},
    ]


def test_induced_rank2_cache_key_depends_on_source_graph_spec() -> None:
    left = induced_rank2_graph_cache_key(
        source_spec=TowerGraphSpec(
            rank=3,
            key_pitch_class=0,
            pitch_min=60,
            pitch_max=69,
            max_step_size=2,
        )
    )
    right = induced_rank2_graph_cache_key(
        source_spec=TowerGraphSpec(
            rank=3,
            key_pitch_class=0,
            pitch_min=60,
            pitch_max=69,
            max_step_size=1,
        )
    )

    assert left != right


def test_induced_rank2_artifact_path_is_deterministic() -> None:
    spec = TowerGraphSpec(rank=3, key_pitch_class=0, pitch_min=60, pitch_max=69, max_step_size=2)

    left = induced_rank2_graph_artifact_path(
        source_spec=spec,
        artifact_root=Path("/tmp/artifacts-b"),
    )
    right = induced_rank2_graph_artifact_path(
        source_spec=spec,
        artifact_root=Path("/tmp/artifacts-b"),
    )

    assert left == right
    assert left.name.endswith(".json")


def test_write_induced_rank2_graph_artifact_persists_payload(tmp_path: Path) -> None:
    spec = TowerGraphSpec(rank=3, key_pitch_class=0, pitch_min=60, pitch_max=69, max_step_size=2)

    artifact_path = write_induced_rank2_graph_artifact(
        source_spec=spec,
        artifact_root=tmp_path,
    )

    assert artifact_path.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "induced_rank2_graph"
    assert payload["artifact_schema_version"] == 2
    assert payload["rank3_legality_contract_version"] == "tower.rank3_legality.v1"
    assert payload["node_image"] == [[60, 67], [60, 68], [62, 69]]
