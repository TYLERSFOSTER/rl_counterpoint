"""Termination and truncation helpers for counterpoint environments."""

from __future__ import annotations


def is_max_step_truncated(*, step_index: int, max_steps: int) -> bool:
    """Return True iff the episode has reached its configured step limit."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    return step_index >= max_steps
