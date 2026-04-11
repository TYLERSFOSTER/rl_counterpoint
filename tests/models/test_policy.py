"""Tests for the first StepDelta policy contract."""

from __future__ import annotations

import pytest
import torch

from rl_counterpoint.models.policy import (
    EncodedTimedChordWindow,
    StepDeltaPolicy,
    SymbolicChordEncoder,
    TransformerStepDeltaPolicy,
    chord_to_unicode_sequence,
    encode_timed_chord_window,
    midi_to_unicode_note_name,
    observation_to_tensor,
    timed_chord_window_to_strings,
    tonic_meter_to_string,
)
from rl_counterpoint.envs.observation import TimedChordWindow


class DummyTextEmbedder:
    """Deterministic test double for symbolic text embedding."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_text(self, text: str) -> torch.Tensor:
        self.calls.append(text)
        base = float(len(text))
        return torch.tensor([base, base + 1.0], dtype=torch.float32)


class ContentAwareDummyTextEmbedder:
    """Deterministic test double that changes embeddings when text changes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_text(self, text: str) -> torch.Tensor:
        self.calls.append(text)
        base = float(sum(ord(ch) for ch in text) % 1000)
        return torch.tensor([base, base + 1.0], dtype=torch.float32)


def test_observation_to_tensor_preserves_chord_state_values() -> None:
    """A raw ChordState is converted to a float tensor in voice order."""
    tensor = observation_to_tensor((3, 6, 10))

    assert tensor.dtype == torch.float32
    assert tensor.shape == (3,)
    assert torch.equal(tensor, torch.tensor([3.0, 6.0, 10.0]))


def test_midi_to_unicode_note_name_uses_standard_midi_octaves() -> None:
    """MIDI note values map to the project's unicode-style note strings."""
    assert midi_to_unicode_note_name(60) == "C_4"
    assert midi_to_unicode_note_name(69) == "A_4"
    assert midi_to_unicode_note_name(72) == "C_5"


def test_chord_to_unicode_sequence_renders_symbolic_chord_text() -> None:
    """Chord states are rendered as symbolic note-name sequences."""
    assert chord_to_unicode_sequence((60, 64, 67)) == "[C_4, E_4, G_4]"


def test_tonic_meter_to_string_renders_context_token() -> None:
    """Tonic and meter are rendered as the shared context-conditioning string."""
    assert tonic_meter_to_string(tonic=60, measure_size=4) == "tonic=C_4 meter=4/4"
    assert (
        tonic_meter_to_string(tonic=60, measure_size=4, target_root_octave=4)
        == "tonic=C_4 meter=4/4 target_root_octave=4"
    )


def test_symbolic_chord_encoder_uses_pad_and_adds_context_embedding() -> None:
    """Timed event encoding adds tonic/meter context to chord-or-PAD embeddings."""
    embedder = DummyTextEmbedder()
    encoder = SymbolicChordEncoder(text_embedder=embedder)
    chord_text = "[C_4, E_4, G_4]"
    context_text = "tonic=C_4 meter=4/4"

    timed_event = encoder.encode_timed_event(chord=(60, 64, 67), tonic=60, measure_size=4)
    padded_event = encoder.encode_timed_event(chord=None, tonic=60, measure_size=4)

    assert embedder.calls[0] == chord_text
    assert embedder.calls[1] == context_text
    assert embedder.calls[2] == "PAD"
    assert embedder.calls[3] == context_text
    expected_timed_event = torch.tensor(
        [len(chord_text) + len(context_text), len(chord_text) + len(context_text) + 2.0]
    )
    expected_padded_event = torch.tensor(
        [len("PAD") + len(context_text), len("PAD") + len(context_text) + 2.0]
    )
    assert torch.equal(timed_event, expected_timed_event)
    assert torch.equal(padded_event, expected_padded_event)


def test_symbolic_chord_encoder_includes_target_root_octave_in_context() -> None:
    """Target octave changes the shared context token and encoded event."""
    embedder = ContentAwareDummyTextEmbedder()
    encoder = SymbolicChordEncoder(text_embedder=embedder)

    encoded = encoder.encode_timed_event(
        chord=(60, 64, 67),
        tonic=60,
        measure_size=4,
        target_root_octave=4,
    )

    assert embedder.calls[0] == "[C_4, E_4, G_4]"
    assert embedder.calls[1] == "tonic=C_4 meter=4/4 target_root_octave=4"
    expected = embedder.embed_text("[C_4, E_4, G_4]") + embedder.embed_text(
        "tonic=C_4 meter=4/4 target_root_octave=4"
    )
    assert torch.equal(encoded, expected)


def test_timed_chord_window_to_strings_renders_pad_and_chord_tokens() -> None:
    """The canonical string window uses PAD at invalid positions."""
    window = TimedChordWindow(
        chord_sequence=((0, 0), (0, 0), (60, 64), (62, 65)),
        bar_positions=(-1, -1, 0, 1),
        valid_mask=(False, False, True, True),
    )

    assert timed_chord_window_to_strings(window) == (
        "PAD",
        "PAD",
        "[C_4, E_4]",
        "[D_4, F_4]",
    )


def test_encode_timed_chord_window_returns_stacked_embeddings_and_mask() -> None:
    """A timed chord window becomes a transformer-ready tensor plus validity mask."""
    window = TimedChordWindow(
        chord_sequence=((0, 0), (60, 64), (62, 65)),
        bar_positions=(-1, 0, 1),
        valid_mask=(False, True, True),
    )
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())

    encoded = encode_timed_chord_window(
        window=window,
        tonic=60,
        measure_size=4,
        encoder=encoder,
    )

    assert isinstance(encoded, EncodedTimedChordWindow)
    assert encoded.event_embeddings.shape == (3, 2)
    assert torch.equal(encoded.valid_mask, torch.tensor([False, True, True]))


def test_encode_timed_chord_window_changes_when_target_octave_changes() -> None:
    """Different target octaves must produce different transformer inputs."""
    window = TimedChordWindow(
        chord_sequence=((0, 0), (60, 64), (62, 65)),
        bar_positions=(-1, 0, 1),
        valid_mask=(False, True, True),
    )
    encoder = SymbolicChordEncoder(text_embedder=ContentAwareDummyTextEmbedder())

    encoded_target_4 = encode_timed_chord_window(
        window=window,
        tonic=60,
        measure_size=4,
        encoder=encoder,
        target_root_octave=4,
    )
    encoded_target_5 = encode_timed_chord_window(
        window=window,
        tonic=60,
        measure_size=4,
        encoder=encoder,
        target_root_octave=5,
    )

    assert not torch.equal(
        encoded_target_4.event_embeddings,
        encoded_target_5.event_embeddings,
    )


def test_encode_timed_chord_window_rejects_non_pad_invalid_slots() -> None:
    """Invalid positions in a padded window must carry the canonical PAD chord."""
    window = TimedChordWindow(
        chord_sequence=((60, 64), (62, 65)),
        bar_positions=(-1, 1),
        valid_mask=(False, True),
    )
    encoder = SymbolicChordEncoder(text_embedder=DummyTextEmbedder())

    with pytest.raises(ValueError, match="invalid padded positions must use the PAD chord"):
        encode_timed_chord_window(
            window=window,
            tonic=60,
            measure_size=4,
            encoder=encoder,
        )


def test_policy_returns_logits_for_unbatched_observation() -> None:
    """The policy maps one observation to one logits vector."""
    policy = StepDeltaPolicy(observation_dim=2, action_dim=8, hidden_dim=16)
    observation = observation_to_tensor((3, 6))

    logits = policy(observation)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()


def test_policy_returns_logits_for_batched_observations() -> None:
    """The policy also accepts a batch of chord-state observations."""
    policy = StepDeltaPolicy(observation_dim=2, action_dim=8, hidden_dim=16)
    observation_batch = torch.tensor([[3.0, 6.0], [4.0, 7.0]])

    logits = policy(observation_batch)

    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()


def test_policy_rejects_invalid_constructor_dimensions() -> None:
    """Observation, action, and hidden dimensions must all be positive."""
    with pytest.raises(ValueError, match="observation_dim must be at least 1"):
        StepDeltaPolicy(observation_dim=0, action_dim=8)

    with pytest.raises(ValueError, match="action_dim must be at least 1"):
        StepDeltaPolicy(observation_dim=2, action_dim=0)

    with pytest.raises(ValueError, match="hidden_dim must be at least 1"):
        StepDeltaPolicy(observation_dim=2, action_dim=8, hidden_dim=0)


def test_policy_rejects_wrong_observation_shapes() -> None:
    """Forward shape checks protect the fixed observation contract."""
    policy = StepDeltaPolicy(observation_dim=2, action_dim=8)

    with pytest.raises(ValueError, match="unbatched observation has wrong shape"):
        policy(torch.tensor([3.0, 6.0, 10.0]))

    with pytest.raises(ValueError, match="batched observation has wrong shape"):
        policy(torch.tensor([[3.0, 6.0, 10.0]]))

    with pytest.raises(ValueError, match="observation tensor must be rank 1 or 2"):
        policy(torch.tensor([[[3.0, 6.0]]]))


def test_transformer_policy_returns_logits_for_encoded_window() -> None:
    """The transformer policy maps one encoded timed window to StepDelta logits."""
    policy = TransformerStepDeltaPolicy(
        embedding_dim=2,
        action_dim=8,
        max_window_len=4,
        d_model=8,
        num_layers=2,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    encoded_window = EncodedTimedChordWindow(
        event_embeddings=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
        valid_mask=torch.tensor([False, True, True, True]),
    )

    logits = policy(encoded_window)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()


def test_transformer_policy_rejects_invalid_constructor_dimensions() -> None:
    """Transformer policy hyperparameters must define a valid encoder."""
    with pytest.raises(ValueError, match="embedding_dim must be at least 1"):
        TransformerStepDeltaPolicy(embedding_dim=0, action_dim=8, max_window_len=4)

    with pytest.raises(ValueError, match="action_dim must be at least 1"):
        TransformerStepDeltaPolicy(embedding_dim=2, action_dim=0, max_window_len=4)

    with pytest.raises(ValueError, match="max_window_len must be at least 1"):
        TransformerStepDeltaPolicy(embedding_dim=2, action_dim=8, max_window_len=0)

    with pytest.raises(ValueError, match="d_model must be divisible by num_heads"):
        TransformerStepDeltaPolicy(
            embedding_dim=2,
            action_dim=8,
            max_window_len=4,
            d_model=10,
            num_heads=3,
        )


def test_transformer_policy_rejects_malformed_encoded_window() -> None:
    """Forward shape checks protect the encoded timed-window contract."""
    policy = TransformerStepDeltaPolicy(
        embedding_dim=2,
        action_dim=8,
        max_window_len=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="event_embeddings must be rank 2"):
        policy(
            EncodedTimedChordWindow(
                event_embeddings=torch.zeros(1, 2, 2),
                valid_mask=torch.tensor([True, True]),
            )
        )

    with pytest.raises(ValueError, match="valid_mask must be rank 1"):
        policy(
            EncodedTimedChordWindow(
                event_embeddings=torch.zeros(2, 2),
                valid_mask=torch.tensor([[True, True]]),
            )
        )

    with pytest.raises(ValueError, match="must share seq_len"):
        policy(
            EncodedTimedChordWindow(
                event_embeddings=torch.zeros(3, 2),
                valid_mask=torch.tensor([True, True]),
            )
        )

    with pytest.raises(ValueError, match="wrong embedding dimension"):
        policy(
            EncodedTimedChordWindow(
                event_embeddings=torch.zeros(2, 3),
                valid_mask=torch.tensor([True, True]),
            )
        )

    with pytest.raises(ValueError, match="must have dtype bool"):
        policy(
            EncodedTimedChordWindow(
                event_embeddings=torch.zeros(2, 2),
                valid_mask=torch.tensor([1, 1]),
            )
        )

    with pytest.raises(ValueError, match="must contain at least one valid event"):
        policy(
            EncodedTimedChordWindow(
                event_embeddings=torch.zeros(2, 2),
                valid_mask=torch.tensor([False, False]),
            )
        )
