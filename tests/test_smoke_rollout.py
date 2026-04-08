"""Tests for the rollout smoke script wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from rl_counterpoint.algos.rollout import PolicyStepRecord
from rl_counterpoint.envs.observation import TimedChordWindow
from scripts import smoke_rollout

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_print_step_record_reports_compact_step_summary(capsys) -> None:
    """The rollout smoke script prints the essential fields of one step record."""
    step = PolicyStepRecord(
        observation=(3, 6),
        timed_window=TimedChordWindow(
            chord_sequence=((0, 0), (0, 0), (3, 6)),
            bar_positions=(-1, -1, 0),
            valid_mask=(False, False, True),
        ),
        action_index=5,
        step_delta=(0, 1),
        action_mask=(False, True, True),
        logits=torch.tensor([0.1, 0.2, 0.3]),
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"state": (3, 7)},
        next_observation=(3, 7),
    )

    smoke_rollout.print_step_record(0, step)
    output = capsys.readouterr().out

    assert "step 0 observation: (3, 6)" in output
    assert "step 0 window:" in output
    assert "step 0 action_index: 5" in output
    assert "step 0 step_delta: (0, 1)" in output
    assert "step 0 legal_action_count: 2" in output
    assert "step 0 logits_dim: 3" in output
    assert "step 0 reward: 1.0" in output
    assert "step 0 next_observation: (3, 7)" in output


def test_main_runs_rollout_smoke_sequence(capsys) -> None:
    """The rollout smoke script prints an episode-length header and step summaries."""
    smoke_rollout.main()
    output = capsys.readouterr().out

    assert "episode length:" in output
    assert "step 0 observation:" in output
    assert "step 0 window:" in output
    assert "step 0 action_index:" in output
    assert "step 0 reward:" in output


def test_smoke_rollout_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/smoke_rollout.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "episode length:" in result.stdout
    assert "step 0 observation:" in result.stdout
    assert "step 0 window:" in result.stdout
    assert "step 0 action_index:" in result.stdout
    assert "step 0 reward:" in result.stdout
