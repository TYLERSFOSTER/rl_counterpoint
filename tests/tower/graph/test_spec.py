"""Tests for minimal tower graph specification."""

from __future__ import annotations

import pytest

from tower.graph.spec import TowerGraphSpec


def test_default_spec_rank_1_accepted() -> None:
    spec = TowerGraphSpec(rank=1)

    assert spec.rank == 1
    assert spec.key_pitch_class == 0
    assert spec.pitch_min == 0
    assert spec.pitch_max == 127
    assert spec.max_step_size == 4


def test_invalid_rank_rejected() -> None:
    with pytest.raises(ValueError, match="rank must be at least 1"):
        TowerGraphSpec(rank=0)


def test_invalid_pitch_range_rejected() -> None:
    with pytest.raises(ValueError, match="pitch_min must be at least 0"):
        TowerGraphSpec(rank=1, pitch_min=-1)

    with pytest.raises(ValueError, match="pitch_max must be at most 127"):
        TowerGraphSpec(rank=1, pitch_max=128)

    with pytest.raises(ValueError, match="pitch_min must be <= pitch_max"):
        TowerGraphSpec(rank=1, pitch_min=10, pitch_max=9)


def test_invalid_max_step_rejected() -> None:
    with pytest.raises(ValueError, match="max_step_size must be at least 1"):
        TowerGraphSpec(rank=1, max_step_size=0)


def test_invalid_key_pitch_class_rejected() -> None:
    with pytest.raises(ValueError, match="key_pitch_class must be in \\[0, 11\\]"):
        TowerGraphSpec(rank=2, key_pitch_class=12)
