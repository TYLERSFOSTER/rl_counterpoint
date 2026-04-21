"""Write deterministic tower reward probe artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower.reward.probe import (
    DEFAULT_REWARD_PROBE_LINEAGE_ID,
    REWARD_PROBE_CASE_NAMES,
    write_slice_a_reward_probe,
)
from tower.train.checkpoint import DEFAULT_TOWER_ARTIFACT_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the reward probe."""
    parser = argparse.ArgumentParser(
        description="Write deterministic Slice A reward probe artifact.",
    )
    parser.add_argument("--lineage-id", default=DEFAULT_REWARD_PROBE_LINEAGE_ID)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_TOWER_ARTIFACT_ROOT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the reward probe artifact."""
    args = parse_args(argv)
    probe_path = write_slice_a_reward_probe(
        artifact_root=args.artifact_root,
        lineage_id=args.lineage_id,
    )

    print(f"probe_path: {probe_path}")
    print(f"case_count: {len(REWARD_PROBE_CASE_NAMES)}")
    print(f"cases: {','.join(REWARD_PROBE_CASE_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
