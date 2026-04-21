"""Deterministic Reward Expansion Slice A probe artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from tower.reward.context import TowerRewardContext
from tower.reward.factory import build_rank1_reward_fn
from tower.state_action import TowerAction, TowerState
from tower.train.config import ARTIFACT_SCHEMA_VERSION
from tower.train.diagnostics import to_json_compatible
from tower.window import build_window

REWARD_TERM_PROBE_FILENAME = "reward_term_probe.jsonl"
DEFAULT_REWARD_PROBE_LINEAGE_ID = "slice-a-reward-probe"
REWARD_PROBE_CASE_NAMES = (
    "terminal_cadence_success",
    "recent_range_penalty",
    "large_leap_recovery_success",
    "large_leap_recovery_failure",
)


@dataclass(frozen=True)
class RewardProbeCase:
    """One deterministic rank-1 reward context probe case."""

    name: str
    history: tuple[TowerState, ...]
    action: TowerAction
    step_index: int
    measure_size: int = 4
    context_measures: int = 2
    is_final_step: bool = False

    def __post_init__(self) -> None:
        if self.name not in REWARD_PROBE_CASE_NAMES:
            raise ValueError("probe case name is not recognized")
        if not self.history:
            raise ValueError("history must not be empty")
        if len(self.action) != 1:
            raise ValueError("probe action must be rank 1")

    @property
    def source(self) -> TowerState:
        """Return the source state for this case."""
        return self.history[-1]

    @property
    def target(self) -> TowerState:
        """Return the target state for this case."""
        return (self.source[0] + self.action[0],)

    def context(self) -> TowerRewardContext:
        """Build the reward context for this case."""
        return TowerRewardContext(
            rank=1,
            step_index=self.step_index,
            source=self.source,
            target=self.target,
            action=self.action,
            window=build_window(
                history=self.history,
                step_index=self.step_index,
                measure_size=self.measure_size,
                context_measures=self.context_measures,
            ),
            measure_size=self.measure_size,
            is_final_step=self.is_final_step,
        )


def slice_a_reward_probe_cases() -> tuple[RewardProbeCase, ...]:
    """Return the stable Slice A reward probe case inventory."""
    return (
        RewardProbeCase(
            name="terminal_cadence_success",
            history=((60,), (67,)),
            action=(-7,),
            step_index=3,
            context_measures=1,
            is_final_step=True,
        ),
        RewardProbeCase(
            name="recent_range_penalty",
            history=((60,), (73,)),
            action=(1,),
            step_index=1,
            context_measures=1,
        ),
        RewardProbeCase(
            name="large_leap_recovery_success",
            history=((60,), (67,)),
            action=(-2,),
            step_index=1,
            context_measures=1,
        ),
        RewardProbeCase(
            name="large_leap_recovery_failure",
            history=((60,), (67,)),
            action=(2,),
            step_index=1,
            context_measures=1,
        ),
    )


def slice_a_reward_probe_rows(
    *,
    lineage_id: str = DEFAULT_REWARD_PROBE_LINEAGE_ID,
) -> tuple[dict[str, object], ...]:
    """Evaluate Slice A reward probe cases and return JSON-compatible rows."""
    if not isinstance(lineage_id, str):
        raise TypeError("lineage_id must be a string")
    if not lineage_id:
        raise ValueError("lineage_id must not be empty")

    reward_fn = build_rank1_reward_fn()
    rows = []
    for case_index, probe_case in enumerate(slice_a_reward_probe_cases()):
        context = probe_case.context()
        result = reward_fn(context)
        rows.append(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "lineage_id": lineage_id,
                "rank": 1,
                "episode_index": 0,
                "episode_kind": "probe",
                "step_index": case_index,
                "case_name": probe_case.name,
                "source_state": list(context.source),
                "assembled_action": list(context.action),
                "attempted_target_state": list(context.target),
                "realized_next_state": list(context.target),
                "reward": float(result.reward),
                "hard_violation": result.hard_violation,
                "is_terminal_success": result.is_terminal_success,
                "reward_diagnostics": to_json_compatible(
                    result.diagnostics,
                    field_name="reward_diagnostics",
                ),
                "terminated": result.is_terminal_success,
                "truncated": False,
                "outcome": "valid",
            }
        )

    return tuple(rows)


def reward_probe_path(*, artifact_root: Path, lineage_id: str) -> Path:
    """Return the Slice A reward probe artifact path."""
    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    if not isinstance(lineage_id, str):
        raise TypeError("lineage_id must be a string")
    if not lineage_id:
        raise ValueError("lineage_id must not be empty")
    return artifact_root / lineage_id / "rank_1" / REWARD_TERM_PROBE_FILENAME


def write_slice_a_reward_probe(
    *,
    artifact_root: Path,
    lineage_id: str = DEFAULT_REWARD_PROBE_LINEAGE_ID,
) -> Path:
    """Write the deterministic Slice A reward probe JSONL artifact."""
    path = reward_probe_path(artifact_root=artifact_root, lineage_id=lineage_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as probe_file:
        for row in slice_a_reward_probe_rows(lineage_id=lineage_id):
            probe_file.write(json.dumps(row, sort_keys=True) + "\n")
    return path
