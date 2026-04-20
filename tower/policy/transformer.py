"""Transformer-family policies for tower ranks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from tower.policy.base import PolicyOutput
from tower.policy.observation import EncodedTowerWindow
from tower.state_action import validate_rank


@dataclass(frozen=True)
class TowerTransformerPolicyConfig:
    """Validated hyperparameters for one rank-local transformer policy."""

    rank: int
    input_feature_dim: int
    action_dim: int
    max_window_len: int
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        _validate_positive_int(
            self.input_feature_dim,
            field_name="input_feature_dim",
        )
        _validate_positive_int(self.action_dim, field_name="action_dim")
        _validate_positive_int(self.max_window_len, field_name="max_window_len")
        _validate_positive_int(self.d_model, field_name="d_model")
        _validate_positive_int(self.num_layers, field_name="num_layers")
        _validate_positive_int(self.num_heads, field_name="num_heads")
        _validate_positive_int(self.ff_dim, field_name="ff_dim")
        if not isinstance(self.dropout, int | float):
            raise TypeError("dropout must be a number")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


class SinusoidalPositionalEncoding(nn.Module):
    """Standard absolute positional encoding for batch-first transformer inputs."""

    def __init__(
        self,
        *,
        d_model: int,
        max_len: int,
    ) -> None:
        super().__init__()
        _validate_positive_int(d_model, field_name="d_model")
        _validate_positive_int(max_len, field_name="max_len")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        positional_encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            positional_encoding[:, 1::2] = torch.cos(
                position * div_term[: positional_encoding[:, 1::2].shape[1]]
            )

        self.register_buffer(
            "positional_encoding",
            positional_encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add absolute positional encoding to a rank-3 batch-first tensor."""
        if x.ndim != 3:
            raise ValueError("x must be rank 3 [batch, seq_len, d_model]")
        sequence_length = x.shape[1]
        if sequence_length > self.positional_encoding.shape[1]:
            raise ValueError("sequence length exceeds positional encoding capacity")
        return x + self.positional_encoding[:, :sequence_length]


class TowerTransformerPolicy(nn.Module):
    """Transformer encoder policy over an encoded rank-local tower window."""

    def __init__(self, config: TowerTransformerPolicyConfig) -> None:
        super().__init__()
        if not isinstance(config, TowerTransformerPolicyConfig):
            raise TypeError("config must be a TowerTransformerPolicyConfig")

        self.config = config
        self.rank = config.rank
        self.input_projection = nn.Linear(config.input_feature_dim, config.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_window_len,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.action_dim),
        )

    def forward(self, encoded_window: EncodedTowerWindow) -> PolicyOutput:
        """Return active-choice logits for one encoded tower window."""
        if not isinstance(encoded_window, EncodedTowerWindow):
            raise TypeError("encoded_window must be an EncodedTowerWindow")
        if encoded_window.rank != self.rank:
            raise ValueError("encoded window rank must match policy rank")

        event_features = encoded_window.event_features
        valid_mask = encoded_window.valid_mask
        if event_features.shape[1] != self.config.input_feature_dim:
            raise ValueError("encoded window has wrong input feature dimension")
        if event_features.shape[0] > self.config.max_window_len:
            raise ValueError("encoded window exceeds max_window_len")

        x = self.input_projection(event_features.unsqueeze(0))
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=(~valid_mask).unsqueeze(0))
        final_index = int(valid_mask.nonzero(as_tuple=False)[-1].item())
        logits = self.output_head(x[0, final_index])

        return PolicyOutput(
            logits=logits,
            diagnostics={
                "rank": self.rank,
                "final_index": final_index,
                "policy": "tower_transformer",
            },
        )


def _validate_positive_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")
