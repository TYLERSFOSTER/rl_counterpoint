"""Tests for tower transformer-family policy configuration."""

from __future__ import annotations

import pytest
import torch

from tower.policy.base import PolicyOutput
from tower.policy.observation import EncodedTowerWindow
from tower.policy.transformer import (
    IndexedSinusoidalPositionalEncoding,
    SinusoidalPositionalEncoding,
    TowerTransformerPolicy,
    TowerTransformerPolicyConfig,
)


def test_transformer_policy_config_accepts_minimal_valid_config() -> None:
    config = TowerTransformerPolicyConfig(
        rank=1,
        input_feature_dim=3,
        action_dim=5,
        max_window_len=8,
        d_model=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    )

    assert config.rank == 1
    assert config.input_feature_dim == 3
    assert config.action_dim == 5
    assert config.max_window_len == 8
    assert config.d_model == 16
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.ff_dim == 32
    assert config.dropout == 0.0


def test_transformer_policy_config_supports_rank_2() -> None:
    config = TowerTransformerPolicyConfig(
        rank=2,
        input_feature_dim=4,
        action_dim=7,
        max_window_len=8,
    )

    assert config.rank == 2
    assert config.input_feature_dim == 4
    assert config.action_dim == 7


@pytest.mark.parametrize(
    ("field_name", "kwargs", "message"),
    (
        ("rank", {"rank": 0}, "rank must be at least 1"),
        ("input_feature_dim", {"input_feature_dim": 0}, "input_feature_dim must be at least 1"),
        ("action_dim", {"action_dim": 0}, "action_dim must be at least 1"),
        ("max_window_len", {"max_window_len": 0}, "max_window_len must be at least 1"),
        ("d_model", {"d_model": 0}, "d_model must be at least 1"),
        ("num_layers", {"num_layers": 0}, "num_layers must be at least 1"),
        ("num_heads", {"num_heads": 0}, "num_heads must be at least 1"),
        ("ff_dim", {"ff_dim": 0}, "ff_dim must be at least 1"),
    ),
)
def test_transformer_policy_config_rejects_non_positive_integer_fields(
    field_name: str,
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "rank": 1,
        "input_feature_dim": 3,
        "action_dim": 5,
        "max_window_len": 8,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 32,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        TowerTransformerPolicyConfig(**values)

    assert field_name in values


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"input_feature_dim": 1.5}, "input_feature_dim must be an int"),
        ({"action_dim": 1.5}, "action_dim must be an int"),
        ({"max_window_len": 1.5}, "max_window_len must be an int"),
        ({"d_model": 1.5}, "d_model must be an int"),
        ({"num_layers": 1.5}, "num_layers must be an int"),
        ({"num_heads": 1.5}, "num_heads must be an int"),
        ({"ff_dim": 1.5}, "ff_dim must be an int"),
    ),
)
def test_transformer_policy_config_rejects_non_integer_fields(
    kwargs: dict[str, float],
    message: str,
) -> None:
    values = {
        "rank": 1,
        "input_feature_dim": 3,
        "action_dim": 5,
        "max_window_len": 8,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 32,
    }
    values.update(kwargs)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=message):
        TowerTransformerPolicyConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("dropout", (-0.1, 1.0))
def test_transformer_policy_config_rejects_dropout_outside_half_open_range(
    dropout: float,
) -> None:
    with pytest.raises(ValueError, match=r"dropout must be in \[0.0, 1.0\)"):
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=8,
            dropout=dropout,
        )


def test_transformer_policy_config_rejects_non_numeric_dropout() -> None:
    with pytest.raises(TypeError, match="dropout must be a number"):
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=8,
            dropout="0.1",  # type: ignore[arg-type]
        )


def test_transformer_policy_config_requires_d_model_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="d_model must be divisible by num_heads"):
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=8,
            d_model=10,
            num_heads=4,
        )


def test_transformer_policy_rejects_non_config_constructor_argument() -> None:
    with pytest.raises(
        TypeError,
        match="config must be a TowerTransformerPolicyConfig",
    ):
        TowerTransformerPolicy(config={"rank": 1})  # type: ignore[arg-type]


def test_transformer_policy_returns_policy_output_for_rank_1_window() -> None:
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    )
    encoded_window = EncodedTowerWindow(
        event_features=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [60.0, 0.0, 1.0],
                [62.0, 1.0, 1.0],
            ]
        ),
        valid_mask=torch.tensor([False, True, True]),
        bar_positions=torch.tensor([-1, 0, 1]),
        rank=1,
    )

    output = policy(encoded_window)

    assert isinstance(output, PolicyOutput)
    assert output.logits.shape == (5,)
    assert torch.isfinite(output.logits).all()
    assert output.diagnostics["rank"] == 1
    assert output.diagnostics["final_index"] == 2


def test_transformer_policy_returns_policy_output_for_rank_2_window() -> None:
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=2,
            input_feature_dim=4,
            action_dim=7,
            max_window_len=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    )
    encoded_window = EncodedTowerWindow(
        event_features=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [60.0, 64.0, 0.0, 1.0],
            ]
        ),
        valid_mask=torch.tensor([False, True]),
        bar_positions=torch.tensor([-1, 0]),
        rank=2,
    )

    output = policy(encoded_window)

    assert output.logits.shape == (7,)
    assert output.diagnostics["rank"] == 2
    assert output.diagnostics["final_index"] == 1


def test_transformer_policy_uses_final_valid_event_not_last_sequence_position() -> None:
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    )
    encoded_window = EncodedTowerWindow(
        event_features=torch.tensor(
            [
                [60.0, 0.0, 1.0],
                [62.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        valid_mask=torch.tensor([True, True, False]),
        bar_positions=torch.tensor([0, 1, -1]),
        rank=1,
    )

    output = policy(encoded_window)

    assert output.diagnostics["final_index"] == 1


def test_transformer_policy_rejects_non_encoded_window() -> None:
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=4,
        )
    )

    with pytest.raises(TypeError, match="encoded_window must be an EncodedTowerWindow"):
        policy(torch.ones((2, 3)))  # type: ignore[arg-type]


def test_transformer_policy_rejects_rank_mismatch() -> None:
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=4,
        )
    )
    encoded_window = EncodedTowerWindow(
        event_features=torch.ones((2, 3)),
        valid_mask=torch.tensor([True, True]),
        bar_positions=torch.tensor([0, 1]),
        rank=2,
    )

    with pytest.raises(ValueError, match="encoded window rank must match"):
        policy(encoded_window)


def test_transformer_policy_rejects_feature_dimension_mismatch() -> None:
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=4,
        )
    )
    encoded_window = EncodedTowerWindow(
        event_features=torch.ones((2, 4)),
        valid_mask=torch.tensor([True, True]),
        bar_positions=torch.tensor([0, 1]),
        rank=1,
    )

    with pytest.raises(ValueError, match="wrong input feature dimension"):
        policy(encoded_window)


def test_transformer_policy_rejects_overlong_encoded_window() -> None:
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=2,
        )
    )
    encoded_window = EncodedTowerWindow(
        event_features=torch.ones((3, 3)),
        valid_mask=torch.tensor([True, True, True]),
        bar_positions=torch.tensor([0, 1, 2]),
        rank=1,
    )

    with pytest.raises(ValueError, match="encoded window exceeds max_window_len"):
        policy(encoded_window)


def test_sinusoidal_positional_encoding_rejects_wrong_input_rank() -> None:
    positional_encoding = SinusoidalPositionalEncoding(d_model=8, max_len=4)

    with pytest.raises(ValueError, match="x must be rank 3"):
        positional_encoding(torch.ones((2, 8)))


def test_indexed_sinusoidal_positional_encoding_rejects_wrong_input_rank() -> None:
    positional_encoding = IndexedSinusoidalPositionalEncoding(d_model=8)

    with pytest.raises(ValueError, match="positions must be rank 2"):
        positional_encoding(torch.ones((4,), dtype=torch.int64))


def test_transformer_policy_uses_episode_step_indices_when_present() -> None:
    torch.manual_seed(0)
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    )
    base_kwargs = dict(
        event_features=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [60.0, 0.0, 1.0],
                [62.0, 1.0, 1.0],
            ]
        ),
        valid_mask=torch.tensor([False, True, True]),
        bar_positions=torch.tensor([-1, 0, 1]),
        rank=1,
    )
    early = EncodedTowerWindow(
        **base_kwargs,
        episode_step_indices=torch.tensor([-1, 0, 1]),
    )
    late = EncodedTowerWindow(
        **base_kwargs,
        episode_step_indices=torch.tensor([-1, 8, 9]),
    )

    early_output = policy(early)
    late_output = policy(late)

    assert not torch.allclose(early_output.logits, late_output.logits)


def test_transformer_policy_uses_frontier_distance_additively() -> None:
    torch.manual_seed(0)
    policy = TowerTransformerPolicy(
        TowerTransformerPolicyConfig(
            rank=1,
            input_feature_dim=3,
            action_dim=5,
            max_window_len=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
        )
    )
    event_features = torch.tensor(
        [
            [60.0, 0.0, 1.0],
            [62.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    early_frontier = EncodedTowerWindow(
        event_features=event_features,
        valid_mask=torch.tensor([True, False, False]),
        bar_positions=torch.tensor([0, -1, -1]),
        episode_step_indices=torch.tensor([4, -1, -1]),
        rank=1,
    )
    late_frontier = EncodedTowerWindow(
        event_features=event_features,
        valid_mask=torch.tensor([True, True, False]),
        bar_positions=torch.tensor([0, 1, -1]),
        episode_step_indices=torch.tensor([4, 5, -1]),
        rank=1,
    )

    early_output = policy(early_frontier)
    late_output = policy(late_frontier)

    assert not torch.allclose(early_output.logits, late_output.logits)
