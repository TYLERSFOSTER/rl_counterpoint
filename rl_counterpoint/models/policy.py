"""Policy modules for fixed-lattice StepDelta action selection."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Protocol
from urllib import request

import torch
from torch import Tensor, nn

from rl_counterpoint.envs.observation import TimedChordWindow, pad_chord
from rl_counterpoint.graph.state_space import ChordState
from rl_counterpoint.music.pitch import midi_to_unicode_note_name
from rl_counterpoint.music.render import chord_to_unicode_sequence, tonic_meter_to_string


class TextEmbedder(Protocol):
    """Minimal embedding interface for symbolic policy encoders."""

    def embed_text(self, text: str) -> Tensor:
        """Return an embedding tensor for one text input."""

@dataclass(frozen=True)
class OpenAITextEmbedder:
    """Embed symbolic text through the OpenAI embeddings API."""

    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1/embeddings"

    def embed_text(self, text: str) -> Tensor:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAITextEmbedder")

        payload = json.dumps({"input": text, "model": self.model}).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))

        embedding = body["data"][0]["embedding"]
        return torch.tensor(embedding, dtype=torch.float32)


@dataclass(frozen=True)
class SymbolicChordEncoder:
    """Encode chord symbols and shared tonic/meter context into model vectors."""

    text_embedder: TextEmbedder

    def encode_chord(self, chord: ChordState) -> Tensor:
        return self.text_embedder.embed_text(chord_to_unicode_sequence(chord))

    def encode_pad(self) -> Tensor:
        return self.text_embedder.embed_text("PAD")

    def encode_context(self, *, tonic: int, measure_size: int) -> Tensor:
        return self.encode_context_with_target(
            tonic=tonic,
            measure_size=measure_size,
            target_root_octave=None,
        )

    def encode_context_with_target(
        self,
        *,
        tonic: int,
        measure_size: int,
        target_root_octave: int | None,
    ) -> Tensor:
        return self.text_embedder.embed_text(
            tonic_meter_to_string(
                tonic=tonic,
                measure_size=measure_size,
                target_root_octave=target_root_octave,
            )
        )

    def encode_timed_event(
        self,
        *,
        chord: ChordState | None,
        tonic: int,
        measure_size: int,
        target_root_octave: int | None = None,
    ) -> Tensor:
        chord_embedding = self.encode_pad() if chord is None else self.encode_chord(chord)
        context_embedding = self.encode_context_with_target(
            tonic=tonic,
            measure_size=measure_size,
            target_root_octave=target_root_octave,
        )
        if chord_embedding.shape != context_embedding.shape:
            raise ValueError("chord and context embeddings must have the same shape")

        return chord_embedding + context_embedding


@dataclass(frozen=True)
class EncodedTimedChordWindow:
    """Transformer-ready tensor contract derived from a timed chord window."""

    event_embeddings: Tensor
    valid_mask: Tensor


def timed_chord_window_to_strings(window: TimedChordWindow) -> tuple[str, ...]:
    """Render a timed chord window as a sequence of symbolic chord/PAD strings."""
    if not window.chord_sequence:
        raise ValueError("window.chord_sequence must not be empty")

    chord_size = len(window.chord_sequence[0])
    pad = pad_chord(chord_size=chord_size)
    return tuple(
        "PAD" if not is_valid else chord_to_unicode_sequence(chord)
        for chord, is_valid in zip(window.chord_sequence, window.valid_mask, strict=True)
        if not (not is_valid and chord != pad)
    )


def encode_timed_chord_window(
    *,
    window: TimedChordWindow,
    tonic: int,
    measure_size: int,
    encoder: SymbolicChordEncoder,
    target_root_octave: int | None = None,
) -> EncodedTimedChordWindow:
    """Encode a padded timed chord window into the transformer input tensor contract."""
    if not window.chord_sequence:
        raise ValueError("window.chord_sequence must not be empty")

    if not (
        len(window.chord_sequence) == len(window.bar_positions) == len(window.valid_mask)
    ):
        raise ValueError("window fields must have the same length")

    chord_size = len(window.chord_sequence[0])
    pad = pad_chord(chord_size=chord_size)
    event_embeddings = []

    for chord, is_valid in zip(window.chord_sequence, window.valid_mask, strict=True):
        if not is_valid and chord != pad:
            raise ValueError("invalid padded positions must use the PAD chord")

        event_embeddings.append(
            encoder.encode_timed_event(
                chord=chord if is_valid else None,
                tonic=tonic,
                measure_size=measure_size,
                target_root_octave=target_root_octave,
            )
        )

    return EncodedTimedChordWindow(
        event_embeddings=torch.stack(event_embeddings),
        valid_mask=torch.tensor(window.valid_mask, dtype=torch.bool),
    )


def observation_to_tensor(observation: ChordState) -> Tensor:
    """Convert a raw ChordState observation into a float tensor."""
    return torch.tensor(observation, dtype=torch.float32)


class StepDeltaPolicy(nn.Module):
    """Tiny policy that maps a chord-state observation to action logits."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if observation_dim < 1:
            raise ValueError("observation_dim must be at least 1")
        if action_dim < 1:
            raise ValueError("action_dim must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """Return logits over the fixed StepDelta action lattice."""
        if observation.ndim == 1:
            if observation.shape[0] != self.observation_dim:
                raise ValueError("unbatched observation has wrong shape")
            return self.network(observation)

        if observation.ndim == 2:
            if observation.shape[1] != self.observation_dim:
                raise ValueError("batched observation has wrong shape")
            return self.network(observation)

        raise ValueError("observation tensor must be rank 1 or 2")


class SinusoidalPositionalEncoding(nn.Module):
    """Standard absolute positional encoding for transformer inputs."""

    def __init__(
        self,
        *,
        d_model: int,
        max_len: int,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be at least 1")
        if max_len < 1:
            raise ValueError("max_len must be at least 1")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add absolute positional encoding to a batch-first sequence tensor."""
        if x.ndim != 3:
            raise ValueError("x must be rank 3 [batch, seq_len, d_model]")
        seq_len = x.shape[1]
        if seq_len > self.pe.shape[1]:
            raise ValueError("sequence length exceeds positional encoding capacity")

        return x + self.pe[:, :seq_len]


class TransformerStepDeltaPolicy(nn.Module):
    """Transformer encoder policy over encoded timed chord windows."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        action_dim: int,
        max_window_len: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if action_dim < 1:
            raise ValueError("action_dim must be at least 1")
        if max_window_len < 1:
            raise ValueError("max_window_len must be at least 1")
        if d_model < 1:
            raise ValueError("d_model must be at least 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if ff_dim < 1:
            raise ValueError("ff_dim must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.max_window_len = max_window_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.input_projection = nn.Linear(embedding_dim, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_window_len,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, encoded_window: EncodedTimedChordWindow) -> Tensor:
        """Return StepDelta logits from one encoded timed chord window."""
        event_embeddings = encoded_window.event_embeddings
        valid_mask = encoded_window.valid_mask

        if event_embeddings.ndim != 2:
            raise ValueError("event_embeddings must be rank 2 [seq_len, embedding_dim]")
        if valid_mask.ndim != 1:
            raise ValueError("valid_mask must be rank 1 [seq_len]")
        if event_embeddings.shape[0] != valid_mask.shape[0]:
            raise ValueError("event_embeddings and valid_mask must share seq_len")
        if event_embeddings.shape[1] != self.embedding_dim:
            raise ValueError("event_embeddings have wrong embedding dimension")
        if event_embeddings.shape[0] > self.max_window_len:
            raise ValueError("encoded window exceeds max_window_len")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must have dtype bool")
        if not bool(valid_mask.any()):
            raise ValueError("encoded window must contain at least one valid event")

        x = self.input_projection(event_embeddings.unsqueeze(0))
        x = self.positional_encoding(x)
        padding_mask = (~valid_mask).unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        final_index = int(valid_mask.nonzero(as_tuple=False)[-1].item())
        final_state = x[0, final_index]
        return self.output_head(final_state)
