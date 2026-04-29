"""Induced lower-rank graph artifacts from projected higher-rank graphs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from tower.graph.actions import action_space
from tower.graph.legality import is_valid_state, is_valid_transition
from tower.graph.projection import project_state
from tower.graph.spec import TowerGraphSpec
from tower.state_action import apply_action

INDUCED_GRAPH_ARTIFACT_SCHEMA_VERSION: Final[int] = 2
INDUCED_RANK1_FROM_RANK2_KIND: Final[str] = "induced_rank1_graph"
INDUCED_RANK2_FROM_RANK3_KIND: Final[str] = "induced_rank2_graph"
PROJECTION_CONVENTION_VERSION: Final[str] = "tower.project_tuple.v1"
RANK2_LEGALITY_CONTRACT_VERSION: Final[str] = "tower.rank2_legality.v5"
RANK3_LEGALITY_CONTRACT_VERSION: Final[str] = "tower.rank3_legality.v2"


def build_induced_rank1_graph_payload(
    *,
    source_spec: TowerGraphSpec,
) -> dict[str, object]:
    """Return the rank-1 graph induced by projecting the pruned rank-2 graph."""
    if source_spec.rank != 2:
        raise ValueError("source_spec.rank must be 2")
    return _build_induced_lower_rank_graph_payload(
        source_spec=source_spec,
        kind=INDUCED_RANK1_FROM_RANK2_KIND,
        legality_contract_key="rank2_legality_contract_version",
        legality_contract_version=RANK2_LEGALITY_CONTRACT_VERSION,
    )


def build_induced_rank2_graph_payload(
    *,
    source_spec: TowerGraphSpec,
) -> dict[str, object]:
    """Return the rank-2 graph induced by projecting the pruned rank-3 graph."""
    if source_spec.rank != 3:
        raise ValueError("source_spec.rank must be 3")
    return _build_induced_lower_rank_graph_payload(
        source_spec=source_spec,
        kind=INDUCED_RANK2_FROM_RANK3_KIND,
        legality_contract_key="rank3_legality_contract_version",
        legality_contract_version=RANK3_LEGALITY_CONTRACT_VERSION,
    )



def induced_rank1_graph_cache_key(*, source_spec: TowerGraphSpec) -> str:
    """Return the deterministic cache key for one rank-2 source graph spec."""
    return _induced_graph_cache_key(
        source_spec=source_spec,
        legality_contract_key="rank2_legality_contract_version",
        legality_contract_version=RANK2_LEGALITY_CONTRACT_VERSION,
    )


def induced_rank1_graph_artifact_path(
    *,
    source_spec: TowerGraphSpec,
    artifact_root: Path,
) -> Path:
    """Return the deterministic artifact path for one induced-rank1 graph."""
    return (
        artifact_root
        / "derived_graphs"
        / "induced_rank1_from_rank2"
        / f"{induced_rank1_graph_cache_key(source_spec=source_spec)}.json"
    )


def induced_rank2_graph_cache_key(*, source_spec: TowerGraphSpec) -> str:
    """Return the deterministic cache key for one rank-3 source graph spec."""
    if source_spec.rank != 3:
        raise ValueError("source_spec.rank must be 3")
    return _induced_graph_cache_key(
        source_spec=source_spec,
        legality_contract_key="rank3_legality_contract_version",
        legality_contract_version=RANK3_LEGALITY_CONTRACT_VERSION,
    )


def induced_rank2_graph_artifact_path(
    *,
    source_spec: TowerGraphSpec,
    artifact_root: Path,
) -> Path:
    """Return the deterministic artifact path for one induced-rank2 graph."""
    if source_spec.rank != 3:
        raise ValueError("source_spec.rank must be 3")
    return (
        artifact_root
        / "derived_graphs"
        / "induced_rank2_from_rank3"
        / f"{induced_rank2_graph_cache_key(source_spec=source_spec)}.json"
    )


def write_induced_rank1_graph_artifact(
    *,
    source_spec: TowerGraphSpec,
    artifact_root: Path,
) -> Path:
    """Persist one induced-rank1 graph artifact and return its path."""
    artifact_path = induced_rank1_graph_artifact_path(
        source_spec=source_spec,
        artifact_root=artifact_root,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_induced_rank1_graph_payload(source_spec=source_spec)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def write_induced_rank2_graph_artifact(
    *,
    source_spec: TowerGraphSpec,
    artifact_root: Path,
) -> Path:
    """Persist one induced-rank2 graph artifact and return its path."""
    if source_spec.rank != 3:
        raise ValueError("source_spec.rank must be 3")
    artifact_path = induced_rank2_graph_artifact_path(
        source_spec=source_spec,
        artifact_root=artifact_root,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_induced_rank2_graph_payload(source_spec=source_spec)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def read_induced_rank1_graph_artifact(path: Path) -> dict[str, object]:
    """Read one induced-rank1 graph artifact payload from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("induced rank1 graph artifact must contain an object")
    return payload


def _enumerate_valid_states(*, spec: TowerGraphSpec) -> tuple[tuple[int, ...], ...]:
    states = []
    if spec.rank == 2:
        for lower_pitch in range(spec.pitch_min, spec.pitch_max + 1):
            for upper_pitch in range(lower_pitch + 1, spec.pitch_max + 1):
                state = (lower_pitch, upper_pitch)
                if is_valid_state(state, spec):
                    states.append(state)
        return tuple(states)

    if spec.rank == 3:
        for lower_pitch in range(spec.pitch_min, spec.pitch_max + 1):
            for middle_pitch in range(lower_pitch + 1, spec.pitch_max + 1):
                for upper_pitch in range(middle_pitch + 1, spec.pitch_max + 1):
                    state = (lower_pitch, middle_pitch, upper_pitch)
                    if is_valid_state(state, spec):
                        states.append(state)
        return tuple(states)

    raise ValueError("state enumeration currently requires rank 2 or rank 3")


def _enumerate_valid_edges(
    *,
    spec: TowerGraphSpec,
    states: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if spec.rank not in {2, 3}:
        raise ValueError("edge enumeration currently requires rank 2 or rank 3")
    edges = []
    for source in states:
        for action in action_space(rank=spec.rank, max_step_size=spec.max_step_size):
            if not is_valid_transition(source, action, spec):
                continue
            target = apply_action(source, action)
            edges.append((source, target))
    return tuple(edges)


def _build_induced_lower_rank_graph_payload(
    *,
    source_spec: TowerGraphSpec,
    kind: str,
    legality_contract_key: str,
    legality_contract_version: str,
) -> dict[str, object]:
    source_nodes = _enumerate_valid_states(spec=source_spec)
    source_edges = _enumerate_valid_edges(spec=source_spec, states=source_nodes)
    node_image = tuple(sorted({project_state(state) for state in source_nodes}))
    edge_image = tuple(
        sorted(
            {
                (projected_source, projected_target)
                for source, target in source_edges
                for projected_source, projected_target in [
                    (project_state(source), project_state(target))
                ]
                if projected_source != projected_target
            }
        )
    )
    target_graph_spec_base = TowerGraphSpec(
        rank=source_spec.rank - 1,
        pitch_min=source_spec.pitch_min,
        pitch_max=source_spec.pitch_max,
        max_step_size=source_spec.max_step_size,
    )
    source_graph_spec_payload = {
        "rank": source_spec.rank,
        "key_pitch_class": source_spec.key_pitch_class,
        "pitch_min": source_spec.pitch_min,
        "pitch_max": source_spec.pitch_max,
        "max_step_size": source_spec.max_step_size,
    }
    target_graph_spec_payload = {
        "rank": target_graph_spec_base.rank,
        "pitch_min": target_graph_spec_base.pitch_min,
        "pitch_max": target_graph_spec_base.pitch_max,
        "max_step_size": target_graph_spec_base.max_step_size,
    }
    return {
        "artifact_schema_version": INDUCED_GRAPH_ARTIFACT_SCHEMA_VERSION,
        "kind": kind,
        "source_rank": source_spec.rank,
        "target_rank": source_spec.rank - 1,
        "projection_convention_version": PROJECTION_CONVENTION_VERSION,
        legality_contract_key: legality_contract_version,
        "source_graph_spec": source_graph_spec_payload,
        "target_graph_spec_base": target_graph_spec_payload,
        "construction_mode": "enumerated",
        "node_count": len(node_image),
        "edge_count": len(edge_image),
        "node_image": [list(state) for state in node_image],
        "edge_image": [
            {"source": list(source), "target": list(target)}
            for source, target in edge_image
        ],
    }


def _induced_graph_cache_key(
    *,
    source_spec: TowerGraphSpec,
    legality_contract_key: str,
    legality_contract_version: str,
) -> str:
    key_payload = {
        "artifact_schema_version": INDUCED_GRAPH_ARTIFACT_SCHEMA_VERSION,
        "projection_convention_version": PROJECTION_CONVENTION_VERSION,
        legality_contract_key: legality_contract_version,
        "source_graph_spec": {
            "rank": source_spec.rank,
            "key_pitch_class": source_spec.key_pitch_class,
            "pitch_min": source_spec.pitch_min,
            "pitch_max": source_spec.pitch_max,
            "max_step_size": source_spec.max_step_size,
            "induced_node_image": (
                None
                if source_spec.induced_node_image is None
                else [list(state) for state in sorted(source_spec.induced_node_image)]
            ),
            "induced_edge_image": (
                None
                if source_spec.induced_edge_image is None
                else [
                    {"source": list(source), "target": list(target)}
                    for source, target in sorted(source_spec.induced_edge_image)
                ]
            ),
        },
    }
    rendered = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    return sha256(rendered.encode("utf-8")).hexdigest()
