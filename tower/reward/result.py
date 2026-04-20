"""Structured reward outputs for tower rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TowerRewardResult:
    """Scalar reward output plus terminal and diagnostic flags."""

    reward: float
    hard_violation: bool = False
    is_terminal_success: bool = False
    diagnostics: Mapping[str, object] = field(default_factory=dict)
