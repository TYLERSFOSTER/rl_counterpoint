"""Import boundary tests for the tower project."""

from __future__ import annotations

from pathlib import Path


def test_tower_source_does_not_import_frozen_legacy_project() -> None:
    project_root = Path(__file__).parents[2]
    tower_root = project_root / "tower"

    offenders = []
    for source_path in tower_root.rglob("*.py"):
        text = source_path.read_text()
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if (
                stripped.startswith("import rl_counterpoint")
                or stripped.startswith("from rl_counterpoint")
            ):
                offenders.append(f"{source_path.relative_to(project_root)}:{line_number}")

    assert offenders == []
