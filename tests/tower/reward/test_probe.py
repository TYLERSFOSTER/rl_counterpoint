"""Tests for deterministic Slice A reward probe artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tower.reward.probe import (
    REWARD_PROBE_CASE_NAMES,
    RewardProbeCase,
    reward_probe_path,
    slice_a_reward_probe_cases,
    slice_a_reward_probe_rows,
    write_slice_a_reward_probe,
)


def term_diagnostics(row: dict[str, object], term_index: int) -> dict[str, object]:
    diagnostics = row["reward_diagnostics"]
    assert isinstance(diagnostics, dict)
    terms = diagnostics["terms"]
    assert isinstance(terms, list)
    term = terms[term_index]
    assert isinstance(term, dict)
    term_diagnostics = term["diagnostics"]
    assert isinstance(term_diagnostics, dict)
    return term_diagnostics


def test_slice_a_reward_probe_cases_are_stable() -> None:
    cases = slice_a_reward_probe_cases()

    assert tuple(probe_case.name for probe_case in cases) == REWARD_PROBE_CASE_NAMES
    assert len(cases) == 4


def test_terminal_cadence_success_probe_row_records_composed_terms() -> None:
    row = slice_a_reward_probe_rows(lineage_id="probe")[0]

    assert row["case_name"] == "terminal_cadence_success"
    assert row["reward"] == 10.5
    assert row["is_terminal_success"] is True
    assert row["terminated"] is True
    assert row["assembled_action"] == [-7]
    cadence = term_diagnostics(row, 0)["cadence"]
    recovery = term_diagnostics(row, 2)["large_leap_recovery"]
    target_octave = term_diagnostics(row, 3)["target_octave_distance"]
    assert cadence["reason"] == "success"
    assert recovery["reason"] == "failed_recovery"
    assert recovery["current_action"] == -7
    assert target_octave["octave_distance"] == 0


def test_recent_range_penalty_probe_row_records_penalty() -> None:
    row = slice_a_reward_probe_rows(lineage_id="probe")[1]

    assert row["case_name"] == "recent_range_penalty"
    assert row["reward"] == -1.0
    cadence = term_diagnostics(row, 0)["cadence"]
    range_diagnostics = term_diagnostics(row, 1)["recent_melodic_range"]
    recovery = term_diagnostics(row, 2)["large_leap_recovery"]
    target_octave = term_diagnostics(row, 3)["target_octave_distance"]
    assert cadence["reason"] == "not_final_step"
    assert range_diagnostics["observed_range"] == 13
    assert range_diagnostics["penalty_applied"] is True
    assert range_diagnostics["reason"] == "range_exceeded"
    assert recovery["reason"] == "failed_recovery"
    assert target_octave["octave_distance"] == 1


def test_large_leap_recovery_success_probe_row_records_reward() -> None:
    row = slice_a_reward_probe_rows(lineage_id="probe")[2]

    assert row["case_name"] == "large_leap_recovery_success"
    assert row["reward"] == 1.5
    recovery = term_diagnostics(row, 2)["large_leap_recovery"]
    target_octave = term_diagnostics(row, 3)["target_octave_distance"]
    assert recovery["previous_interval"] == 7
    assert recovery["current_action"] == -2
    assert recovery["opposite_direction"] is True
    assert recovery["success"] is True
    assert recovery["reason"] == "recovered"
    assert target_octave["octave_distance"] == 0


def test_large_leap_recovery_failure_probe_row_records_penalty() -> None:
    row = slice_a_reward_probe_rows(lineage_id="probe")[3]

    assert row["case_name"] == "large_leap_recovery_failure"
    assert row["reward"] == 0.5
    recovery = term_diagnostics(row, 2)["large_leap_recovery"]
    target_octave = term_diagnostics(row, 3)["target_octave_distance"]
    assert recovery["previous_interval"] == 7
    assert recovery["current_action"] == 2
    assert recovery["opposite_direction"] is False
    assert recovery["success"] is False
    assert recovery["reason"] == "failed_recovery"
    assert target_octave["octave_distance"] == 0


def test_reward_probe_path_is_deterministic(tmp_path: Path) -> None:
    assert reward_probe_path(artifact_root=tmp_path, lineage_id="probe") == (
        tmp_path / "probe" / "rank_1" / "reward_term_probe.jsonl"
    )


def test_write_slice_a_reward_probe_emits_jsonl(tmp_path: Path) -> None:
    path = write_slice_a_reward_probe(artifact_root=tmp_path, lineage_id="probe")

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert path == tmp_path / "probe" / "rank_1" / "reward_term_probe.jsonl"
    assert len(rows) == 4
    assert [row["case_name"] for row in rows] == list(REWARD_PROBE_CASE_NAMES)
    assert all(row["episode_kind"] == "probe" for row in rows)


def test_reward_probe_case_rejects_unknown_case_name() -> None:
    with pytest.raises(ValueError, match="probe case name is not recognized"):
        RewardProbeCase(
            name="unknown",
            history=((60,),),
            action=(1,),
            step_index=0,
        )
