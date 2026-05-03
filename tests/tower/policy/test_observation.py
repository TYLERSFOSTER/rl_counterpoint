"""Tests for tower policy observation contracts."""

from __future__ import annotations

import pytest
import torch

from tower.policy.observation import EncodedTowerWindow, encode_tower_window
from tower.window import TowerWindow, build_window


def test_encoded_tower_window_accepts_valid_rank_local_tensors() -> None:
    event_features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [60.0, 0.0, 1.0],
            [62.0, 1.0, 1.0],
        ]
    )
    valid_mask = torch.tensor([False, True, True])
    bar_positions = torch.tensor([-1, 0, 1])

    encoded = EncodedTowerWindow(
        event_features=event_features,
        valid_mask=valid_mask,
        bar_positions=bar_positions,
        rank=1,
        context={
            "measure_size": 4,
            "key_pitch_class": 0,
            "target_root_octave": 4,
        },
    )

    assert encoded.event_features is event_features
    assert encoded.valid_mask is valid_mask
    assert encoded.bar_positions is bar_positions
    assert encoded.rank == 1
    assert encoded.context["measure_size"] == 4
    assert encoded.context["key_pitch_class"] == 0
    assert encoded.context["target_root_octave"] == 4


def test_encoded_tower_window_supports_rank_2_feature_width() -> None:
    encoded = EncodedTowerWindow(
        event_features=torch.tensor(
            [
                [60.0, 64.0, 0.0],
                [62.0, 65.0, 1.0],
            ]
        ),
        valid_mask=torch.tensor([True, True]),
        bar_positions=torch.tensor([0, 1]),
        rank=2,
    )

    assert encoded.event_features.shape == (2, 3)
    assert encoded.rank == 2


def test_encoded_tower_window_rejects_non_tensor_event_features() -> None:
    with pytest.raises(TypeError, match="event_features must be a torch.Tensor"):
        EncodedTowerWindow(
            event_features=[[60.0]],  # type: ignore[arg-type]
            valid_mask=torch.tensor([True]),
            bar_positions=torch.tensor([0]),
            rank=1,
        )


def test_encoded_tower_window_rejects_non_sequence_event_features() -> None:
    with pytest.raises(ValueError, match="event_features must be rank 2"):
        EncodedTowerWindow(
            event_features=torch.tensor([60.0]),
            valid_mask=torch.tensor([True]),
            bar_positions=torch.tensor([0]),
            rank=1,
        )


def test_encoded_tower_window_rejects_empty_event_sequence() -> None:
    with pytest.raises(ValueError, match="at least one event"):
        EncodedTowerWindow(
            event_features=torch.empty((0, 3)),
            valid_mask=torch.empty((0,), dtype=torch.bool),
            bar_positions=torch.empty((0,), dtype=torch.int64),
            rank=1,
        )


def test_encoded_tower_window_rejects_valid_mask_length_mismatch() -> None:
    with pytest.raises(ValueError, match="valid_mask length must match"):
        EncodedTowerWindow(
            event_features=torch.ones((2, 3)),
            valid_mask=torch.tensor([True]),
            bar_positions=torch.tensor([0, 1]),
            rank=1,
        )


def test_encoded_tower_window_rejects_bar_position_length_mismatch() -> None:
    with pytest.raises(ValueError, match="bar_positions length must match"):
        EncodedTowerWindow(
            event_features=torch.ones((2, 3)),
            valid_mask=torch.tensor([True, True]),
            bar_positions=torch.tensor([0]),
            rank=1,
        )


def test_encoded_tower_window_rejects_non_bool_valid_mask() -> None:
    with pytest.raises(TypeError, match="valid_mask must have dtype torch.bool"):
        EncodedTowerWindow(
            event_features=torch.ones((2, 3)),
            valid_mask=torch.tensor([1, 1]),
            bar_positions=torch.tensor([0, 1]),
            rank=1,
        )


def test_encoded_tower_window_rejects_non_numeric_bar_positions() -> None:
    with pytest.raises(TypeError, match="bar_positions must have numeric dtype"):
        EncodedTowerWindow(
            event_features=torch.ones((2, 3)),
            valid_mask=torch.tensor([True, True]),
            bar_positions=torch.tensor([False, True]),
            rank=1,
        )


def test_encoded_tower_window_rejects_no_valid_events() -> None:
    with pytest.raises(ValueError, match="at least one valid event"):
        EncodedTowerWindow(
            event_features=torch.ones((2, 3)),
            valid_mask=torch.tensor([False, False]),
            bar_positions=torch.tensor([-1, -1]),
            rank=1,
        )


def test_encoded_tower_window_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank must be at least 1"):
        EncodedTowerWindow(
            event_features=torch.ones((1, 3)),
            valid_mask=torch.tensor([True]),
            bar_positions=torch.tensor([0]),
            rank=0,
        )


def test_encoded_tower_window_rejects_non_mapping_context() -> None:
    with pytest.raises(TypeError, match="context must be a mapping"):
        EncodedTowerWindow(
            event_features=torch.ones((1, 3)),
            valid_mask=torch.tensor([True]),
            bar_positions=torch.tensor([0]),
            rank=1,
            context=("measure_size", 4),  # type: ignore[arg-type]
        )


def test_encode_tower_window_encodes_rank_1_states() -> None:
    window = build_window(
        history=((60,), (62,)),
        step_index=1,
        measure_size=4,
        context_measures=1,
    )

    encoded = encode_tower_window(
        window=window,
        measure_size=4,
        key_pitch_class=0,
        target_root_octave=4,
        max_step_size=2,
    )

    assert encoded.rank == 1
    assert torch.allclose(
        encoded.event_features,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.25],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.25],
                [60.0 / 127.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.25],
                [62.0 / 127.0, 1.0 / 3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.25],
            ]
        ),
    )
    assert torch.equal(encoded.valid_mask, torch.tensor([False, False, True, True]))
    assert torch.equal(encoded.bar_positions, torch.tensor([-1, -1, 0, 1]))
    assert encoded.context == {
        "measure_size": 4,
        "key_pitch_class": 0,
        "target_root_octave": 4,
        "max_step_size": 2,
    }


def test_encode_tower_window_changes_features_when_target_octave_changes() -> None:
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    low_target = encode_tower_window(
        window=window,
        measure_size=4,
        target_root_octave=3,
    )
    high_target = encode_tower_window(
        window=window,
        measure_size=4,
        target_root_octave=5,
    )

    assert low_target.event_features.shape == (4, 9)
    assert high_target.event_features.shape == (4, 9)
    assert not torch.equal(low_target.event_features, high_target.event_features)
    assert torch.equal(low_target.event_features[:, 7], torch.full((4,), 0.4))
    assert torch.equal(high_target.event_features[:, 7], torch.full((4,), 0.6))
    assert torch.equal(low_target.event_features[:, 8], torch.full((4,), 0.25))


def test_encode_tower_window_changes_features_when_key_pitch_class_changes() -> None:
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    c_major = encode_tower_window(
        window=window,
        measure_size=4,
        key_pitch_class=0,
    )
    d_major = encode_tower_window(
        window=window,
        measure_size=4,
        key_pitch_class=2,
    )

    assert c_major.event_features.shape == (4, 9)
    assert d_major.event_features.shape == (4, 9)
    assert not torch.equal(c_major.event_features, d_major.event_features)
    assert torch.equal(c_major.event_features[:, 7], torch.full((4,), 0.0))
    assert torch.equal(d_major.event_features[:, 7], torch.full((4,), 2.0 / 11.0))
    assert torch.equal(c_major.event_features[:, 8], torch.full((4,), 0.25))


def test_encode_tower_window_encodes_rank_2_states() -> None:
    window = build_window(
        history=((60, 64), (62, 65)),
        step_index=1,
        measure_size=4,
        context_measures=1,
    )

    encoded = encode_tower_window(window=window, measure_size=4)

    assert encoded.rank == 2
    assert torch.allclose(
        encoded.event_features,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                [60.0 / 127.0, 64.0 / 127.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.25],
                [62.0 / 127.0, 65.0 / 127.0, 1.0 / 3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.25],
            ]
        ),
    )
    assert torch.equal(encoded.valid_mask, torch.tensor([False, False, True, True]))
    assert torch.equal(encoded.bar_positions, torch.tensor([-1, -1, 0, 1]))
    assert encoded.context == {"measure_size": 4}


def test_encode_tower_window_allows_rank_2_pad_state_not_strictly_increasing() -> None:
    window = build_window(
        history=((60, 64),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    encoded = encode_tower_window(window=window, measure_size=4)

    assert encoded.rank == 2
    assert torch.allclose(
        encoded.event_features,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                [60.0 / 127.0, 64.0 / 127.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25],
            ]
        ),
    )
    assert torch.equal(encoded.valid_mask, torch.tensor([False, False, False, True]))


def test_encode_tower_window_includes_episode_step_and_frontier_distance() -> None:
    window = build_window(
        history=((60,), (61,), (62,)),
        step_index=6,
        measure_size=4,
        context_measures=1,
    )

    encoded = encode_tower_window(window=window, measure_size=4)

    assert torch.equal(encoded.event_features[:, 5], torch.tensor([0.0, 4.0, 5.0, 6.0]))
    assert torch.equal(encoded.event_features[:, 6], torch.tensor([0.0, 2.0, 1.0, 0.0]))


def test_encode_tower_window_merges_extra_context() -> None:
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    encoded = encode_tower_window(
        window=window,
        measure_size=4,
        context={
            "lineage_id": "lineage-a",
            "rank_label": "root",
        },
    )

    assert encoded.context == {
        "measure_size": 4,
        "lineage_id": "lineage-a",
        "rank_label": "root",
    }


def test_encode_tower_window_rejects_non_window_input() -> None:
    with pytest.raises(TypeError, match="window must be a TowerWindow"):
        encode_tower_window(  # type: ignore[arg-type]
            window=((60,),),
            measure_size=4,
        )


def test_encode_tower_window_rejects_invalid_measure_size() -> None:
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    with pytest.raises(ValueError, match="measure_size must be at least 1"):
        encode_tower_window(window=window, measure_size=0)


def test_encode_tower_window_rejects_mismatched_window_lengths() -> None:
    window = TowerWindow(
        states=((60,),),
        bar_positions=(0, 1),
        valid_mask=(True,),
    )

    with pytest.raises(ValueError, match="window fields must have the same length"):
        encode_tower_window(window=window, measure_size=4)


def test_encode_tower_window_rejects_rank_mismatch() -> None:
    window = build_window(
        history=((60,),),
        step_index=0,
        measure_size=4,
        context_measures=1,
    )

    with pytest.raises(ValueError, match="state length must match rank"):
        encode_tower_window(window=window, measure_size=4, rank=2)


def test_encode_tower_window_rejects_mismatched_episode_step_indices() -> None:
    window = TowerWindow(
        states=((60,),),
        bar_positions=(0,),
        valid_mask=(True,),
        episode_step_indices=(0, 1),
    )

    with pytest.raises(ValueError, match="window fields must have the same length"):
        encode_tower_window(window=window, measure_size=4)
