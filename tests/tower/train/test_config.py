"""Tests for tower training config contracts."""

from __future__ import annotations

import json

import pytest

from tower.train.config import ARTIFACT_SCHEMA_VERSION, TowerRankConfig


def test_rank_1_config_round_trips_through_json_dict() -> None:
    config = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=100,
        reward_config={"success_reward": 10.0},
        graph_config={"pitch_min": 48, "pitch_max": 72},
        policy_config={"hidden_sizes": [32, 32]},
        training_config={"learning_rate": 1e-3, "gamma": 0.99},
        seed_config={"python": 123},
    )

    payload = config.to_json_dict()
    restored = TowerRankConfig.from_json_dict(payload)

    assert restored == config
    assert payload["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    json.dumps(payload)


def test_rank_2_config_round_trips_with_parent_metadata() -> None:
    config = TowerRankConfig(
        rank=2,
        lineage_id="lineage-a",
        episode_budget=200,
        parent_checkpoint="rank_1/checkpoint_latest.pt",
        parent_sampler_config={"top_m": 3, "temperature": 0.25},
    )

    restored = TowerRankConfig.from_json_dict(config.to_json_dict())

    assert restored.rank == 2
    assert restored.parent_checkpoint == "rank_1/checkpoint_latest.pt"
    assert restored.parent_sampler_config == {"top_m": 3, "temperature": 0.25}


def test_config_from_json_uses_defaults_for_optional_fields() -> None:
    config = TowerRankConfig.from_json_dict(
        {
            "rank": 1,
            "lineage_id": "lineage-a",
            "episode_budget": 10,
        }
    )

    assert config.measure_size == 4
    assert config.context_measures == 2
    assert config.max_step_size == 4
    assert config.parent_checkpoint is None
    assert config.reward_config == {}


def test_config_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank must be at least 1"):
        TowerRankConfig(
            rank=0,
            lineage_id="lineage-a",
            episode_budget=1,
        )


def test_config_rejects_invalid_episode_budget() -> None:
    with pytest.raises(ValueError, match="episode_budget must be at least 1"):
        TowerRankConfig(
            rank=1,
            lineage_id="lineage-a",
            episode_budget=0,
        )


def test_config_rejects_empty_lineage_id() -> None:
    with pytest.raises(ValueError, match="lineage_id must not be empty"):
        TowerRankConfig(
            rank=1,
            lineage_id="",
            episode_budget=1,
        )


def test_config_rejects_non_json_config_value() -> None:
    with pytest.raises(TypeError, match="reward_config.bad must be JSON-compatible"):
        TowerRankConfig(
            rank=1,
            lineage_id="lineage-a",
            episode_budget=1,
            reward_config={"bad": object()},
        )


def test_config_rejects_missing_required_json_field() -> None:
    with pytest.raises(ValueError, match="episode_budget is required"):
        TowerRankConfig.from_json_dict(
            {
                "rank": 1,
                "lineage_id": "lineage-a",
            }
        )


def test_config_payload_is_json_compatible() -> None:
    payload = TowerRankConfig(
        rank=1,
        lineage_id="lineage-a",
        episode_budget=1,
        reward_config={"weights": [1.0, 2.0], "enabled": True, "name": None},
    ).to_json_dict()

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["reward_config"] == {
        "weights": [1.0, 2.0],
        "enabled": True,
        "name": None,
    }
