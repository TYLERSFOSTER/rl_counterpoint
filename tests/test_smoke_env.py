"""Tests for the environment smoke script wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import smoke_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_first_legal_step_delta_returns_first_true_mask_entry() -> None:
    """The smoke script selects the first legal StepDelta by mask order."""
    action_space = ((-1, 0), (0, 1), (1, 1))
    action_mask = (False, True, True)

    assert smoke_env.first_legal_step_delta(action_space, action_mask) == (0, 1)


def test_first_legal_step_delta_raises_when_no_action_is_legal() -> None:
    """The smoke script fails explicitly when the mask has no legal moves."""
    with pytest.raises(RuntimeError, match="no legal StepDelta found"):
        smoke_env.first_legal_step_delta(((-1, 0), (0, 1)), (False, False))


def test_print_info_summary_reports_mask_counts(capsys: pytest.CaptureFixture[str]) -> None:
    """The info summary prints state, step index, and action-mask counts."""
    smoke_env.print_info_summary(
        "reset",
        {
            "state": (3, 6),
            "step_index": 0,
            "action_mask": (True, False, True),
            "has_legal_actions": True,
        },
    )

    output = capsys.readouterr().out

    assert "reset state: (3, 6)" in output
    assert "reset step_index: 0" in output
    assert "reset action_count: 3" in output
    assert "reset legal_action_count: 2" in output
    assert "reset has_legal_actions: True" in output


def test_print_info_summary_rejects_non_tuple_action_mask() -> None:
    """The info summary protects its expected env info shape."""
    with pytest.raises(TypeError, match="action_mask must be a tuple"):
        smoke_env.print_info_summary(
            "reset",
            {
                "state": (3, 6),
                "step_index": 0,
                "action_mask": [True, False],
                "has_legal_actions": True,
            },
        )


def test_main_runs_smoke_sequence(capsys: pytest.CaptureFixture[str]) -> None:
    """The smoke script main prints the reset and one-step env sequence."""
    smoke_env.main()

    output = capsys.readouterr().out

    assert "reset obs:" in output
    assert "chosen StepDelta:" in output
    assert "step obs:" in output
    assert "step reward:" in output
    assert "step valid_action:" in output


def test_smoke_env_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/smoke_env.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "reset obs:" in result.stdout
    assert "chosen StepDelta:" in result.stdout
    assert "step obs:" in result.stdout
    assert "step valid_action:" in result.stdout
