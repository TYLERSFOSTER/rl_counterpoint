"""Training configuration contracts for tower artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from tower.state_action import validate_rank

ARTIFACT_SCHEMA_VERSION = 1
JsonObject = dict[str, object]


@dataclass(frozen=True)
class TowerRankConfig:
    """JSON-friendly configuration payload for one tower training rank."""

    rank: int
    lineage_id: str
    episode_budget: int
    measure_size: int = 4
    context_measures: int = 2
    max_step_size: int = 4
    reward_config: Mapping[str, object] = field(default_factory=dict)
    graph_config: Mapping[str, object] = field(default_factory=dict)
    policy_config: Mapping[str, object] = field(default_factory=dict)
    training_config: Mapping[str, object] = field(default_factory=dict)
    parent_sampler_config: Mapping[str, object] = field(default_factory=dict)
    parent_checkpoint: str | None = None
    seed_config: Mapping[str, object] = field(default_factory=dict)
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        if not isinstance(self.lineage_id, str):
            raise TypeError("lineage_id must be a string")
        if not self.lineage_id:
            raise ValueError("lineage_id must not be empty")
        if self.episode_budget < 1:
            raise ValueError("episode_budget must be at least 1")
        if self.measure_size < 1:
            raise ValueError("measure_size must be at least 1")
        if self.context_measures < 1:
            raise ValueError("context_measures must be at least 1")
        if self.max_step_size < 1:
            raise ValueError("max_step_size must be at least 1")
        if self.artifact_schema_version < 1:
            raise ValueError("artifact_schema_version must be at least 1")
        if self.parent_checkpoint is not None and not isinstance(
            self.parent_checkpoint,
            str,
        ):
            raise TypeError("parent_checkpoint must be a string or None")

        _validate_json_mapping(self.reward_config, field_name="reward_config")
        _validate_json_mapping(self.graph_config, field_name="graph_config")
        _validate_json_mapping(self.policy_config, field_name="policy_config")
        _validate_json_mapping(self.training_config, field_name="training_config")
        _validate_json_mapping(
            self.parent_sampler_config,
            field_name="parent_sampler_config",
        )
        _validate_json_mapping(self.seed_config, field_name="seed_config")

    def to_json_dict(self) -> JsonObject:
        """Return a JSON-compatible dictionary representation."""
        return {
            "rank": self.rank,
            "lineage_id": self.lineage_id,
            "episode_budget": self.episode_budget,
            "measure_size": self.measure_size,
            "context_measures": self.context_measures,
            "max_step_size": self.max_step_size,
            "reward_config": dict(self.reward_config),
            "graph_config": dict(self.graph_config),
            "policy_config": dict(self.policy_config),
            "training_config": dict(self.training_config),
            "parent_sampler_config": dict(self.parent_sampler_config),
            "parent_checkpoint": self.parent_checkpoint,
            "seed_config": dict(self.seed_config),
            "artifact_schema_version": self.artifact_schema_version,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> TowerRankConfig:
        """Build a rank config from a JSON-compatible dictionary."""
        _validate_json_mapping(payload, field_name="payload")
        return cls(
            rank=_required_int(payload, "rank"),
            lineage_id=_required_str(payload, "lineage_id"),
            episode_budget=_required_int(payload, "episode_budget"),
            measure_size=_optional_int(payload, "measure_size", default=4),
            context_measures=_optional_int(payload, "context_measures", default=2),
            max_step_size=_optional_int(payload, "max_step_size", default=4),
            reward_config=_optional_mapping(payload, "reward_config"),
            graph_config=_optional_mapping(payload, "graph_config"),
            policy_config=_optional_mapping(payload, "policy_config"),
            training_config=_optional_mapping(payload, "training_config"),
            parent_sampler_config=_optional_mapping(payload, "parent_sampler_config"),
            parent_checkpoint=_optional_str_or_none(payload, "parent_checkpoint"),
            seed_config=_optional_mapping(payload, "seed_config"),
            artifact_schema_version=_optional_int(
                payload,
                "artifact_schema_version",
                default=ARTIFACT_SCHEMA_VERSION,
            ),
        )


def _required_int(payload: Mapping[str, object], key: str) -> int:
    if key not in payload:
        raise ValueError(f"{key} is required")
    value = payload[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value


def _optional_int(payload: Mapping[str, object], key: str, *, default: int) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    if key not in payload:
        raise ValueError(f"{key} is required")
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_str_or_none(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{key} must be a string or None")


def _optional_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    if key not in payload:
        return {}
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    _validate_json_mapping(value, field_name=key)
    return dict(value)


def _validate_json_mapping(payload: Mapping[str, object], *, field_name: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        _validate_json_value(value, field_name=f"{field_name}.{key}")


def _validate_json_value(value: object, *, field_name: str) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, Mapping):
        _validate_json_mapping(value, field_name=field_name)
        return
    raise TypeError(f"{field_name} must be JSON-compatible")

