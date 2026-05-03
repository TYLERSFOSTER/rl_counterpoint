"""Shared CLI defaults for tower reward diagnostics logging."""

from __future__ import annotations

AUTO_FULL_REWARD_DIAGNOSTICS_STEP_BUDGET = 100_000


def resolve_log_reward_diagnostics(
    *,
    requested: bool | None,
    episode_count: int,
    max_steps: int,
) -> tuple[bool, str]:
    """Resolve per-step training diagnostics logging for tower train scripts."""
    if not isinstance(episode_count, int):
        raise TypeError("episode_count must be an int")
    if episode_count < 1:
        raise ValueError("episode_count must be at least 1")
    if not isinstance(max_steps, int):
        raise TypeError("max_steps must be an int")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if requested is True:
        return True, "full-explicit"
    if requested is False:
        return False, "final-only-explicit"

    projected_training_steps = episode_count * max_steps
    if projected_training_steps <= AUTO_FULL_REWARD_DIAGNOSTICS_STEP_BUDGET:
        return True, "full-auto"
    return False, "final-only-auto"
