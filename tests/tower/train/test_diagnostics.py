"""Tests for tower reward diagnostics artifact serialization."""

from __future__ import annotations

import pytest
import torch

from tower.reward.result import TowerRewardResult
from tower.train.diagnostics import reward_diagnostics_rows, to_json_compatible
from tower.train.trajectory import TRAJECTORY_OUTCOME_VALID, TowerTrajectory, TowerTrajectoryStep
from tower.window import build_window


def make_trajectory(
    *,
    reward_diagnostics: dict[str, object] | None = None,
    rank: int = 1,
) -> TowerTrajectory:
    source_state = (60,) if rank == 1 else (60, 64)
    action = (1,) if rank == 1 else (1, 1)
    target_state = (61,) if rank == 1 else (61, 65)
    return TowerTrajectory(
        steps=(
            TowerTrajectoryStep(
                rank=rank,
                step_index=0,
                source_state=source_state,
                window=build_window(
                    history=(source_state,),
                    step_index=0,
                    measure_size=4,
                    context_measures=1,
                ),
                parent_state=None if rank == 1 else (60,),
                parent_action=None if rank == 1 else (1,),
                active_choice=1,
                assembled_action=action,
                attempted_target_state=target_state,
                realized_next_state=target_state,
                active_logprob=None,
                parent_logprob=None,
                reward=TowerRewardResult(
                    reward=1.25,
                    diagnostics={}
                    if reward_diagnostics is None
                    else reward_diagnostics,
                ),
                terminated=False,
                truncated=True,
                outcome=TRAJECTORY_OUTCOME_VALID,
                diagnostics={"active_choices": (-1, 1)},
            ),
        )
    )


def test_reward_diagnostics_rows_serialize_required_step_fields() -> None:
    trajectory = make_trajectory(
        reward_diagnostics={
            "kind": "rank1_reward",
            "terms": ({"reward": 1.25, "diagnostics": {"reason": "success"}},),
        }
    )

    rows = reward_diagnostics_rows(
        trajectory=trajectory,
        lineage_id="lineage-a",
        episode_index=0,
        episode_kind="training",
    )

    assert rows == (
        {
            "artifact_schema_version": 1,
            "lineage_id": "lineage-a",
            "rank": 1,
            "episode_index": 0,
            "episode_kind": "training",
            "step_index": 0,
            "source_state": [60],
            "assembled_action": [1],
            "attempted_target_state": [61],
            "realized_next_state": [61],
            "reward": 1.25,
            "hard_violation": False,
            "is_terminal_success": False,
            "reward_diagnostics": {
                "kind": "rank1_reward",
                "terms": [
                    {"reward": 1.25, "diagnostics": {"reason": "success"}},
                ],
            },
            "terminated": False,
            "truncated": True,
            "outcome": "valid",
            "active_choice": 1,
            "step_diagnostics": {"active_choices": [-1, 1]},
        },
    )


def test_reward_diagnostics_rows_include_rank_2_parent_fields() -> None:
    rows = reward_diagnostics_rows(
        trajectory=make_trajectory(rank=2),
        lineage_id="lineage-a",
        episode_index=1,
        episode_kind="final_inference",
    )

    assert rows[0]["parent_state"] == [60]
    assert rows[0]["parent_action"] == [1]
    assert rows[0]["episode_kind"] == "final_inference"


def test_reward_diagnostics_rows_reject_invalid_episode_kind() -> None:
    with pytest.raises(ValueError, match="episode_kind must be"):
        reward_diagnostics_rows(
            trajectory=make_trajectory(),
            lineage_id="lineage-a",
            episode_index=0,
            episode_kind="evaluation",  # type: ignore[arg-type]
        )


def test_to_json_compatible_rejects_tensors() -> None:
    with pytest.raises(TypeError, match="must not contain torch.Tensor"):
        to_json_compatible(
            {"bad": torch.tensor(1.0)},
            field_name="diagnostics",
        )
