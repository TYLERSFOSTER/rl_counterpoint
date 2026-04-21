"""Model-observation contracts for rank-local tower policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from tower.state_action import rank_of_state, validate_rank, validate_state
from tower.window import TowerWindow


@dataclass(frozen=True)
class EncodedTowerWindow:
    """Transformer-ready tensor view of a rank-local tower window."""

    event_features: torch.Tensor
    valid_mask: torch.Tensor
    bar_positions: torch.Tensor
    rank: int
    context: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_rank(self.rank)
        _validate_event_features(self.event_features)
        _validate_valid_mask(self.valid_mask)
        _validate_bar_positions(self.bar_positions)
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")

        sequence_length = self.event_features.shape[0]
        if self.valid_mask.shape[0] != sequence_length:
            raise ValueError("valid_mask length must match event_features")
        if self.bar_positions.shape[0] != sequence_length:
            raise ValueError("bar_positions length must match event_features")
        if not bool(self.valid_mask.any()):
            raise ValueError("encoded window must contain at least one valid event")


def encode_tower_window(
    *,
    window: TowerWindow,
    measure_size: int,
    rank: int | None = None,
    key_pitch_class: int | None = None,
    target_root_octave: int | None = None,
    max_step_size: int | None = None,
    context: Mapping[str, object] | None = None,
) -> EncodedTowerWindow:
    """Encode a rank-local tower window as numeric policy features."""
    if not isinstance(window, TowerWindow):
        raise TypeError("window must be a TowerWindow")
    if measure_size < 1:
        raise ValueError("measure_size must be at least 1")
    _validate_window_lengths(window)

    inferred_rank = _infer_window_rank(window)
    encoded_rank = inferred_rank if rank is None else rank
    validate_rank(encoded_rank)
    for state, is_valid in zip(window.states, window.valid_mask, strict=True):
        if is_valid:
            validate_state(state, rank=encoded_rank)
        else:
            _validate_padded_state_shape(state, rank=encoded_rank)

    observation_context = {
        "measure_size": measure_size,
        **({} if key_pitch_class is None else {"key_pitch_class": key_pitch_class}),
        **(
            {}
            if target_root_octave is None
            else {"target_root_octave": target_root_octave}
        ),
        **({} if max_step_size is None else {"max_step_size": max_step_size}),
        **({} if context is None else dict(context)),
    }

    event_features = torch.tensor(window.states, dtype=torch.float32)
    if key_pitch_class is not None:
        key_feature = torch.full(
            (len(window.states), 1),
            float(key_pitch_class),
            dtype=torch.float32,
        )
        event_features = torch.cat((event_features, key_feature), dim=1)
    if target_root_octave is not None:
        target_feature = torch.full(
            (len(window.states), 1),
            float(target_root_octave),
            dtype=torch.float32,
        )
        event_features = torch.cat((event_features, target_feature), dim=1)

    return EncodedTowerWindow(
        event_features=event_features,
        valid_mask=torch.tensor(window.valid_mask, dtype=torch.bool),
        bar_positions=torch.tensor(window.bar_positions, dtype=torch.int64),
        rank=encoded_rank,
        context=observation_context,
    )


def _validate_window_lengths(window: TowerWindow) -> None:
    if not window.states:
        raise ValueError("window.states must not be empty")
    if not (
        len(window.states) == len(window.valid_mask) == len(window.bar_positions)
    ):
        raise ValueError("window fields must have the same length")


def _infer_window_rank(window: TowerWindow) -> int:
    for state, is_valid in zip(window.states, window.valid_mask, strict=True):
        if is_valid:
            return rank_of_state(state)
    raise ValueError("window must contain at least one valid state")


def _validate_padded_state_shape(state: tuple[int, ...], *, rank: int) -> None:
    if not isinstance(state, tuple):
        raise TypeError("padded states must be tuples")
    if len(state) != rank:
        raise ValueError("padded state length must match rank")
    for pitch in state:
        if not isinstance(pitch, int):
            raise TypeError("padded state entries must be ints")
        if pitch < 0 or pitch > 127:
            raise ValueError("padded state entries must be in [0, 127]")


def _validate_event_features(event_features: torch.Tensor) -> None:
    if not isinstance(event_features, torch.Tensor):
        raise TypeError("event_features must be a torch.Tensor")
    if event_features.ndim != 2:
        raise ValueError("event_features must be rank 2 [seq_len, feature_dim]")
    if event_features.shape[0] < 1:
        raise ValueError("event_features must contain at least one event")
    if event_features.shape[1] < 1:
        raise ValueError("event_features must contain at least one feature")


def _validate_valid_mask(valid_mask: torch.Tensor) -> None:
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if valid_mask.ndim != 1:
        raise ValueError("valid_mask must be rank 1 [seq_len]")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have dtype torch.bool")


def _validate_bar_positions(bar_positions: torch.Tensor) -> None:
    if not isinstance(bar_positions, torch.Tensor):
        raise TypeError("bar_positions must be a torch.Tensor")
    if bar_positions.ndim != 1:
        raise ValueError("bar_positions must be rank 1 [seq_len]")
    if not bar_positions.dtype.is_floating_point and not (
        bar_positions.dtype == torch.int8
        or bar_positions.dtype == torch.int16
        or bar_positions.dtype == torch.int32
        or bar_positions.dtype == torch.int64
        or bar_positions.dtype == torch.uint8
    ):
        raise TypeError("bar_positions must have numeric dtype")
