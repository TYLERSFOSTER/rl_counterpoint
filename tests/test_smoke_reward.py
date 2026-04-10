"""Tests for the reward smoke script wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rl_counterpoint.reward.protocol import RewardResult
from scripts import smoke_reward

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_make_example_context_builds_expected_reward_context() -> None:
    """The smoke script builds a deterministic reward context example."""
    context = smoke_reward.make_example_context(step_index=3)

    assert context.step_index == 3
    assert context.measure_size == 4
    assert context.key_pitch_class == 0
    assert context.step_delta == (0, 1, 0)
    assert context.history == ((60, 64, 67),)
    assert context.timed_chord_window is not None
    assert context.timed_chord_window.bar_positions == (-1, 3)


def test_print_reward_summary_reports_compact_reward_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reward smoke summary prints the essential reward diagnostics."""
    smoke_reward.print_reward_summary(
        source=(60, 64, 67),
        target=(60, 64, 67),
        result=RewardResult(
            reward=1.25,
            diagnostics={
                "kind": "strong_beat_consonance",
                "step_index": 0,
                "is_strong_beat": True,
                "applied_beat_weight": 1.0,
                "base_static_consonance_reward": 1.25,
            },
        ),
    )

    output = capsys.readouterr().out

    assert "source: (60, 64, 67)" in output
    assert "target: (60, 64, 67)" in output
    assert "reward: 1.25" in output
    assert "kind: strong_beat_consonance" in output
    assert "step_index: 0" in output
    assert "is_strong_beat: True" in output
    assert "applied_beat_weight: 1.0" in output


def test_main_runs_reward_smoke_sequence(capsys: pytest.CaptureFixture[str]) -> None:
    """The reward smoke main prints both strong- and weak-beat examples."""
    smoke_reward.main()

    output = capsys.readouterr().out

    assert "strong-beat example" in output
    assert "weak-beat example" in output
    assert "kind: strong_beat_consonance" in output
    assert "is_strong_beat: True" in output
    assert "is_strong_beat: False" in output


def test_smoke_reward_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/smoke_reward.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "strong-beat example" in result.stdout
    assert "weak-beat example" in result.stdout
    assert "kind: strong_beat_consonance" in result.stdout
