"""Tests for the graph smoke script wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rl_counterpoint.graph.graph_spec import CounterpointGraphSpec
from rl_counterpoint.graph.actions import step_delta_action_mask, step_delta_action_space
from scripts import smoke_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_first_legal_transition_returns_first_true_mask_entry() -> None:
    """The smoke script selects the first legal transition by mask order."""
    step_delta, target = smoke_graph.first_legal_transition(
        (3, 6),
        ((-1, 0), (0, 1), (1, 1)),
        (False, True, True),
    )

    assert step_delta == (0, 1)
    assert target == (3, 7)


def test_first_legal_transition_raises_when_no_action_is_legal() -> None:
    """The smoke script fails explicitly when the mask has no legal moves."""
    with pytest.raises(RuntimeError, match="no legal StepDelta found"):
        smoke_graph.first_legal_transition((3, 6), ((-1, 0), (0, 1)), (False, False))


def test_print_state_summary_reports_graph_and_mask_facts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The graph smoke summary prints node and action-mask facts."""
    spec = CounterpointGraphSpec(n=2, tonic=60)
    state = (3, 6)
    action_space = step_delta_action_space(n=spec.n, max_step_size=2)
    action_mask = step_delta_action_mask(state, action_space, spec)

    smoke_graph.print_state_summary(
        state=state,
        spec=spec,
        action_space=action_space,
        action_mask=action_mask,
    )

    output = capsys.readouterr().out

    assert "state: (3, 6)" in output
    assert "root_pitch_class: 3" in output
    assert "adjacent_intervals: (3,)" in output
    assert "outer_interval: 3" in output
    assert "is_valid_node: True" in output
    assert "action_count:" in output
    assert "legal_action_count:" in output


def test_print_state_summary_rejects_mismatched_mask_lengths() -> None:
    """The graph smoke summary protects its expected action-space shape."""
    with pytest.raises(
        ValueError,
        match="action_space and action_mask must have the same length",
    ):
        smoke_graph.print_state_summary(
            state=(3, 6),
            spec=CounterpointGraphSpec(n=2, tonic=60),
            action_space=((0, 1),),
            action_mask=(True, False),
        )


def test_main_runs_graph_smoke_sequence(capsys: pytest.CaptureFixture[str]) -> None:
    """The smoke script main prints the node summary and one legal transition."""
    smoke_graph.main()

    output = capsys.readouterr().out

    assert "state: (3, 6)" in output
    assert "adjacent_intervals:" in output
    assert "chosen StepDelta:" in output
    assert "candidate target:" in output
    assert "is_valid_edge: True" in output


def test_smoke_graph_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/smoke_graph.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "state: (3, 6)" in result.stdout
    assert "chosen StepDelta:" in result.stdout
    assert "candidate target:" in result.stdout
    assert "is_valid_edge: True" in result.stdout
