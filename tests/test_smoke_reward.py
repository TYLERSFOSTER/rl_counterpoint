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
    context = smoke_reward.make_example_context(
        step_index=3,
        target_root_octave=4,
        is_final_step=True,
    )

    assert context.step_index == 3
    assert context.measure_size == 4
    assert context.key_pitch_class == 0
    assert context.step_delta == (0, 1, 0)
    assert context.history == ((60, 64, 67),)
    assert context.target_root_octave == 4
    assert context.is_final_step
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
                "kind": "target_root_octave",
                "step_index": 0,
                "root_octave": 4,
                "target_root_octave": 5,
                "octave_distance": 1,
                "distance_reward": 0.5,
                "is_final_step": False,
                "terminal_root_octaves": (),
                "terminal_distances": (),
                "terminal_window_average": 0.0,
                "terminal_bonus": 0.0,
                "terminal_match": False,
            },
        ),
    )

    output = capsys.readouterr().out

    assert "source: (60, 64, 67)" in output
    assert "target: (60, 64, 67)" in output
    assert "reward: 1.25" in output
    assert "kind: target_root_octave" in output
    assert "step_index: 0" in output
    assert "root_octave: 4" in output
    assert "target_root_octave: 5" in output
    assert "octave_distance: 1" in output
    assert "distance_reward: 0.5" in output
    assert "is_final_step: False" in output
    assert "terminal_root_octaves: ()" in output
    assert "terminal_distances: ()" in output
    assert "terminal_window_average: 0.0" in output
    assert "terminal_root_octaves:" in output
    assert "terminal_distances:" in output
    assert "terminal_window_average:" in output
    assert "terminal_bonus: 0.0" in output
    assert "terminal_match: False" in output


def test_main_runs_reward_smoke_sequence(capsys: pytest.CaptureFixture[str]) -> None:
    """The reward smoke main prints both shaping and final-hit examples."""
    smoke_reward.main()

    output = capsys.readouterr().out

    assert "distance-shaping example" in output
    assert "final-hit example" in output
    assert "kind: target_root_octave" in output
    assert "terminal_match: False" in output
    assert "terminal_window_average:" in output
    assert "terminal_bonus:" in output


def test_smoke_reward_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/smoke_reward.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "distance-shaping example" in result.stdout
    assert "final-hit example" in result.stdout
    assert "kind: target_root_octave" in result.stdout
