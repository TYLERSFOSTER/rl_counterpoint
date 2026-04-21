"""Tests for deterministic Slice A reward probe script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import tower_reward_probe

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_tower_reward_probe_parse_args_defaults() -> None:
    args = tower_reward_probe.parse_args([])

    assert args.lineage_id == "slice-a-reward-probe"
    assert args.artifact_root == PROJECT_ROOT / "artifacts" / "tower"


def test_tower_reward_probe_main_writes_probe_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = tower_reward_probe.main(
        [
            "--lineage-id",
            "probe",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    probe_path = tmp_path / "probe" / "rank_1" / "reward_term_probe.jsonl"
    assert exit_code == 0
    assert f"probe_path: {probe_path}" in output
    assert "case_count: 4" in output
    assert "terminal_cadence_success" in output
    assert probe_path.exists()
    assert len(probe_path.read_text().splitlines()) == 4


def test_tower_reward_probe_script_runs_by_file_path(tmp_path: Path) -> None:
    script_path = PROJECT_ROOT / "scripts" / "tower_reward_probe.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--lineage-id",
            "probe",
            "--artifact-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    probe_path = tmp_path / "probe" / "rank_1" / "reward_term_probe.jsonl"
    assert f"probe_path: {probe_path}" in result.stdout
    assert "case_count: 4" in result.stdout
    assert probe_path.exists()
