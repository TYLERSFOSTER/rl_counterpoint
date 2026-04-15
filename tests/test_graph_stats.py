"""Tests for the graph stats script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rl_counterpoint.graph.actions import step_delta_action_mask
from scripts import graph_stats
from scripts.train_reinforce import TrainConfig, build_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legal_out_star_size_matches_mask_count() -> None:
    """The helper counts legal outgoing step deltas from one node."""
    config = TrainConfig(voice_count=2, max_step_size=2)
    env = build_env(config)
    state = env.initial_state
    expected = sum(step_delta_action_mask(state, env.action_space, env.graph_spec))

    out_star_size = graph_stats.legal_out_star_size(
        state,
        action_space=env.action_space,
        graph_spec=env.graph_spec,
    )

    assert out_star_size == expected


def test_average_out_star_size_is_positive() -> None:
    """The graph stats script reports a positive average out-degree."""
    average = graph_stats.average_out_star_size(
        TrainConfig(voice_count=2, max_step_size=2)
    )

    assert average > 0.0


def test_rough_goal_step_count_is_positive() -> None:
    """The heuristic root-motion depth estimate is a positive step count."""
    rough_steps = graph_stats.rough_goal_step_count(
        TrainConfig(voice_count=2, max_step_size=2)
    )

    assert rough_steps >= 1


def test_branch_growth_estimate_is_larger_than_average_branch() -> None:
    """The b^d estimate exceeds b when the heuristic depth is greater than 1."""
    config = TrainConfig(voice_count=2, max_step_size=2)
    average = graph_stats.average_out_star_size(config)
    growth = graph_stats.branch_growth_estimate(config)

    assert growth > average


def test_main_prints_average_out_star_size(capsys) -> None:
    """The script main prints the current average out-star size."""
    graph_stats.main()

    output = capsys.readouterr().out

    assert "voice_count:" in output
    assert "max_step_size:" in output
    assert "average_out_star_size:" in output
    assert "rough_goal_step_count:" in output
    assert "heuristic_depth_d:" in output
    assert "branch_growth_estimate_b_to_d:" in output


def test_graph_stats_script_runs_by_file_path() -> None:
    """Direct script execution can import the project package from repo root."""
    result = subprocess.run(
        [sys.executable, "scripts/graph_stats.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "average_out_star_size:" in result.stdout
    assert "branch_growth_estimate_b_to_d:" in result.stdout
