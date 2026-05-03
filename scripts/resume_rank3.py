"""Resume an interrupted rank-3 tower lineage from checkpoint."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._tower_resume import resume_rank3_lineage
from tower.train.checkpoint import TowerArtifactPaths
from tower.train.lifecycle import write_run_completion, write_run_failure, write_run_heartbeat


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume an interrupted rank-3 lineage.")
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tower",
    )
    parser.add_argument("--initial-pedal-pitch", type=int, default=62)
    parser.add_argument("--initial-middle-pitch", type=int, default=65)
    parser.add_argument("--initial-top-pitch", type=int, default=69)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exception_log_path = args.artifact_root / "logs" / f"{args.lineage_id}-rank3-resume.exception.log"
    write_run_heartbeat(
        args.artifact_root / args.lineage_id,
        lineage_id=args.lineage_id,
        stage="rank3-resume",
        status="running",
        extra={"argv": [] if argv is None else list(argv)},
    )
    try:
        print(f"resuming rank3 lineage: {args.lineage_id}", flush=True)
        resume_rank3_lineage(
            lineage_id=args.lineage_id,
            artifact_root=args.artifact_root,
            initial_state=(
                args.initial_pedal_pitch,
                args.initial_middle_pitch,
                args.initial_top_pitch,
            ),
        )
        paths = TowerArtifactPaths(
            lineage_id=args.lineage_id,
            rank=3,
            artifact_root=args.artifact_root,
        )
        write_run_completion(
            paths,
            lineage_id=args.lineage_id,
            stage="rank3-resume",
            summary={
                "run_dir": paths.rank_dir.as_posix(),
                "checkpoint_latest_path": paths.checkpoint_latest_path.as_posix(),
            },
        )
        return 0
    except Exception as exc:
        exception_log_path.parent.mkdir(parents=True, exist_ok=True)
        exception_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_run_failure(
            args.artifact_root / args.lineage_id,
            lineage_id=args.lineage_id,
            stage="rank3-resume",
            error_type=type(exc).__name__,
            error_message=str(exc),
            exception_log=exception_log_path.as_posix(),
        )
        print(f"exception log: {exception_log_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
