"""Tests for the tiny explicit REINFORCE training entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import train_reinforce

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_reinforce_main_prints_training_stats(capsys) -> None:
    """The training entrypoint runs a tiny explicit REINFORCE loop."""
    train_reinforce.main()
    output = capsys.readouterr().out

    assert "episode 0 return:" in output
    assert "episode 0 length:" in output
    assert "episode 0 loss:" in output


def test_train_reinforce_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/train_reinforce.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "episode 0 return:" in result.stdout
    assert "episode 0 length:" in result.stdout
    assert "episode 0 loss:" in result.stdout
